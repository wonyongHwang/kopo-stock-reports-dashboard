# -*- coding: utf-8 -*-
"""
Analyst Ranking & Evidence Dashboard (Horizon-only)
- 상단 Top Analysts: 행 클릭(체크박스 없이)으로 선택 → 하단에 상세 근거표 표시
- 30/60일 성과 섹션 제거 (요청사항)
- Firestore 구조는 evaluations/by_analyst 저장 스키마에 맞춤
- 브로커명 정규화(공백 제거 등) 저장·조회 전 구간에 적용
"""

import math
import datetime as dt
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import pytz

from google.cloud import firestore
from st_aggrid import DataReturnMode

import re

# -----------------------------
# 정규화 & 유틸
# -----------------------------
def normalize_broker_name(name: str) -> str:
    """브로커명 공백/특수문자 제거, 표준화"""
    if not name:
        return ""
    name = re.sub(r"\s+", "", name)           # 모든 공백 제거
    name = re.sub(r"[\(\)\[\]]", "", name)    # 괄호류 제거
    return name.strip()

def _norm_url(x: str) -> str:
    if not x:
        return ""
    s = str(x).strip()
    if s.startswith(("http://", "https://")):
        return s
    return "https://" + s

KST = pytz.timezone("Asia/Seoul")

def format_ts(ts, tz=KST):
    """Firestore Timestamp 또는 문자열/None → 보기용 문자열"""
    if not ts:
        return "-"
    try:
        dt_utc = ts if isinstance(ts, dt.datetime) else ts.to_datetime()
    except Exception:
        try:
            dt_utc = dt.datetime.fromisoformat(str(ts))
        except Exception:
            return str(ts)
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=dt.timezone.utc)
    return dt_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")

# --- (추가) 선택 행 인덱스 안전 추출 헬퍼 ---
def _extract_selected_idx_from_aggrid(grid_ret) -> int | None:
    sel = grid_ret.get("selected_rows", [])
    # DataFrame 케이스
    if isinstance(sel, pd.DataFrame):
        if sel.empty:
            return None
        return int(sel.iloc[0]["_row"])
    # list[dict] 케이스
    if isinstance(sel, list):
        if not sel:
            return None
        row0 = sel[0]
        try:
            return int(row0.get("_row"))
        except AttributeError:
            return int(row0["_row"])
    return None

# ====== 선택 UI: streamlit-aggrid 유무에 따른 분기 ======
_AGGRID_AVAILABLE = True
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode
except Exception:
    _AGGRID_AVAILABLE = False

# -----------------------------
# 전역/헬퍼
# -----------------------------
EXCLUDED_BROKERS = {"한국IR협의회"}  # 제외할 증권사명 집합

def _parse_report_date(val) -> dt.date | None:
    """report_date가 문자열/타임스탬프/None 등 섞여 있어도 안전하게 date로 변환"""
    try:
        if hasattr(val, "to_datetime"):
            val = val.to_datetime()
        if isinstance(val, dt.datetime):
            return val.date()
        s = (str(val) or "").strip()
        if not s:
            return None
        # 우선 ISO yyyy-mm-dd 우선
        if len(s) == 10 and s[4] == "-" and s[7] == "-":
            return dt.date.fromisoformat(s)
        # 느슨한 포맷도 수용
        return dt.date.fromisoformat(pd.to_datetime(s).date().isoformat())
    except Exception:
        return None

def filter_docs_by_date(docs: list[dict], date_from: dt.date | None, date_to: dt.date | None) -> list[dict]:
    """report_date 기준으로 한 번 더 클라이언트 필터(랭킹 전용 안전장치)"""
    if not (date_from or date_to):
        return docs
    out = []
    for x in docs:
        d = _parse_report_date(x.get("report_date"))
        if d is None:
            continue
        if date_from and d < date_from:
            continue
        if date_to and d > date_to:
            continue
        out.append(x)
    return out


def is_allowed_broker(broker: str) -> bool:
    # 비교도 정규화값으로
    return normalize_broker_name(broker) not in {normalize_broker_name(b) for b in EXCLUDED_BROKERS}

# -----------------------------
# Firestore Client & Caches
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_db():
    return firestore.Client()

@st.cache_data(show_spinner=False, ttl=60)
def load_broker_list() -> List[str]:
    """by_analyst에서 broker만 모아 정규화 후 유니크 반환"""
    db = get_db()
    q = db.collection_group("by_analyst").select(["broker"]).limit(5000)
    s = set()
    try:
        for d in q.stream():
            b = (d.to_dict() or {}).get("broker")
            b = normalize_broker_name(b)
            if b and is_allowed_broker(b):
                s.add(b)
    except Exception:
        pass
    return sorted(s)

@st.cache_data(show_spinner=False, ttl=60)
def load_analyst_docs(min_date: dt.date | None, max_date: dt.date | None,
                      brokers: List[str] | None,
                      limit: int = 3000) -> List[Dict[str, Any]]:
    """
    by_analyst 문서 로드 (필요한 필터만 서버 측 where, 나머지는 클라이언트 필터)
    brokers: 정규화된 이름 리스트를 기대
    """
    db = get_db()
    q = db.collection_group("by_analyst")
    try:
        if min_date: q = q.where("report_date", ">=", min_date.isoformat())
        if max_date: q = q.where("report_date", "<=", max_date.isoformat())
        q = q.limit(limit)
        docs = list(q.stream())
    except Exception:
        # 인덱스 이슈 시 범위를 줄여서라도 읽기
        q = db.collection_group("by_analyst").limit(limit)
        docs = list(q.stream())

    out: List[Dict[str, Any]] = []
    broker_set = {normalize_broker_name(b) for b in (brokers or [])} or None
    for d in docs:
        x = d.to_dict() or {}
        # 정규화
        broker_raw = (x.get("broker") or "").strip()
        broker_norm = normalize_broker_name(broker_raw)
        # 제외
        if not is_allowed_broker(broker_norm):
            continue
        # 선택 필터
        if broker_set and broker_norm not in broker_set:
            continue

        x["broker"] = broker_norm   # 정규화값으로 통일
        # 구형 호환: points_total 없으면 horizon 합으로 대체
        if x.get("points_total") is None and x.get("horizons"):
            x["points_total"] = sum(float(h.get("points_total_this_horizon") or 0.0) for h in x.get("horizons") or [])
        out.append(x)
    return out

@st.cache_data(show_spinner=False, ttl=60)
def load_detail_for_analyst(analyst_name: str, broker: str,
                            date_from: dt.date | None, date_to: dt.date | None,
                            limit: int = 800) -> List[Dict[str, Any]]:
    """
    특정 애널리스트(이름+브로커)의 by_analyst 문서 목록 로드
    - 정규화된 broker로 1차 조회
    - (과거데이터 호환) 정규화 전 broker가 다르면 2차 조회 후 병합
    """
    db = get_db()
    out: List[Dict[str, Any]] = []

    broker_norm = normalize_broker_name(broker)

    def _query(b: str) -> List[Dict[str, Any]]:
        q = (db.collection_group("by_analyst")
               .where("broker", "==", normalize_broker_name(b))
               .where("analyst_name", "==", analyst_name))
        try:
            if date_from: q = q.where("report_date", ">=", date_from.isoformat())
            if date_to:   q = q.where("report_date", "<=", date_to.isoformat())
        except Exception:
            pass
        q = q.limit(limit)
        docs = [z.to_dict() or {} for z in q.stream()]
        # 브로커 정규화 보정
        for z in docs:
            z["broker"] = normalize_broker_name(z.get("broker"))
        return docs

    # 1차: 정규화 값으로
    out.extend(_query(broker_norm))
    # 2차: 비정규화가 다를 경우(과거 문서 호환)
    if broker != broker_norm:
        out.extend(_query(broker))  # 중복 가능

    # 중복 제거(report_id, pdf_url 기준 우선)
    seen: set[Tuple[str, str]] = set()
    uniq: List[Dict[str, Any]] = []
    for r in out:
        key = (str(r.get("report_id","")), str(r.get("pdf_url","")))
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r)

    uniq.sort(key=lambda r: r.get("report_date",""), reverse=True)
    return uniq

# -----------------------------
# Helpers (표현/집계)
# -----------------------------
def _price_stats(price_sample: list[dict]) -> tuple[float|None, float|None, float|None]:
    if not price_sample:
        return None, None, None
    closes = [float(p.get("close")) for p in price_sample if p.get("close") is not None]
    if not closes:
        return None, None, None
    max_c = max(closes)
    min_c = min(closes)
    last_c = closes[-1]
    return max_c, min_c, last_c

def normalize_name(s: str) -> str:
    return (s or "").strip()

def _sign(x: float | None) -> int | None:
    if x is None:
        return None
    if x > 0: return 1
    if x < 0: return -1
    return 0

def _desired_from_horizon(h: Dict[str, Any]) -> int | None:
    dtg = h.get("direction_target") or {}
    desired = dtg.get("desired")
    try:
        return int(desired) if desired is not None else None
    except Exception:
        return None

def format_pos_label(desired: int | None, rating_norm: str | None) -> str:
    r = (rating_norm or "").lower()
    if desired == 1 or r == "buy":  return "상승(buy)"
    if desired == -1 or r == "sell": return "하락(sell)"
    return "중립(hold)"

def horizon_to_rows(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    by_analyst 한 문서(= 한 리포트) → 여러 horizon 근거 행으로 펼침
    """
    rows = []
    horizons = rec.get("horizons") or []
    rating_norm = rec.get("rating_norm") or rec.get("rating")
    target_price = rec.get("target_price")

    for h in horizons:
        desired = _desired_from_horizon(h)
        ret = h.get("return_pct", None)  # start_close 대비 end_close 수익률
        actual_sign = _sign(ret)
        correct = None
        if desired is not None and actual_sign is not None:
            if desired == 0:
                correct = (actual_sign == 0)
            else:
                correct = (desired == actual_sign)

        price_sample = h.get("price_sample") or []
        max_c, min_c, last_c = _price_stats(price_sample)

        # 목표가 대비 수익률 (마지막 종가 기준)
        target_return_pct = None
        if target_price and target_price not in ("", 0, 0.0) and last_c:
            try:
                target_return_pct = round((float(last_c) - float(target_price)) / float(target_price) * 100, 3)
            except Exception:
                target_return_pct = None

        # 목표가 대비 최대 도달률 (최고가 기준)
        max_reach_pct = None
        if target_price and target_price not in ("", 0, 0.0) and max_c:
            try:
                max_reach_pct = round((float(max_c) / float(target_price) - 1) * 100, 3)
            except Exception:
                max_reach_pct = None

        rows.append({
            "보고서일": rec.get("report_date",""),
            "horizon(일)": h.get("horizon_days"),
            "예측포지션": format_pos_label(desired, rating_norm),
            "레이팅": rating_norm,
            "예측목표가": target_price,

            # 기존 수익률 (리포트 시작 대비)
            "리포트일에 매수했을경우 수익률%": None if ret is None else round(ret*100, 3),

            # 목표가 대비
            "목표가대비수익률%": target_return_pct,
            "목표가대비최대도달%": max_reach_pct,

            "방향정답": None if correct is None else ("✅" if correct else "❌"),
            "방향점수": h.get("direction_point", 0.0),
            "목표근접점수": h.get("target_proximity_point", 0.0),
            "목표가HIT": h.get("hit_target_anytime", None),

            "구간최고종가": max_c,
            "구간최저종가": min_c,
            "구간마지막종가": last_c,

            "종목": rec.get("stock",""),
            "티커": rec.get("ticker",""),
            "제목": rec.get("title",""),
            "리포트PDF": rec.get("pdf_url","") or rec.get("report_url",""),
            "상세페이지": rec.get("detail_url",""),
            "리포트총점": rec.get("points_total", 0.0),
        })
    return rows

def make_rank_table(docs: list[dict], metric: str = "avg", min_reports: int = 2) -> pd.DataFrame:
    from collections import defaultdict
    agg = defaultdict(lambda: {"sum":0.0, "n":0, "name":"", "broker":"", "first":None, "last":None})

    if not docs:
        return pd.DataFrame(columns=["RankScore","Reports","Analyst","Broker","FirstReport","LastReport"])

    # 집계
    for x in docs:
        name = (x.get("analyst_name") or "").strip() or "(unknown)"
        broker = normalize_broker_name((x.get("broker") or "").strip())
        try:
            pts = float(x.get("points_total") or 0.0)
        except Exception:
            pts = 0.0
        rdate = (x.get("report_date") or "").strip()

        key = f"{name}|{broker}"
        agg[key]["sum"] += pts
        agg[key]["n"] += 1
        agg[key]["name"] = name
        agg[key]["broker"] = broker
        if rdate:
            if not agg[key]["first"] or rdate < agg[key]["first"]:
                agg[key]["first"] = rdate
            if not agg[key]["last"] or rdate > agg[key]["last"]:
                agg[key]["last"]  = rdate

    rows = []
    for _, v in agg.items():
        if v["n"] < min_reports:
            continue
        score = (v["sum"]/v["n"]) if metric == "avg" else v["sum"]
        rows.append({
            "RankScore": round(score, 4),
            "Reports": v["n"],
            "Analyst": v["name"],
            "Broker": v["broker"],
            "FirstReport": v["first"] or "",
            "LastReport": v["last"] or "",
        })

    if not rows:
        return pd.DataFrame(columns=["RankScore","Reports","Analyst","Broker","FirstReport","LastReport"])

    df = pd.DataFrame(rows)
    if not {"RankScore","Reports"}.issubset(df.columns):
        return df.reset_index(drop=True)
    return df.sort_values(["RankScore","Reports"], ascending=[False, False]).reset_index(drop=True)

def get_last_updated_from_docs(docs):
    latest = None
    for d in docs:
        ts = d.get("updated_at")
        if ts is None:
            continue
        try:
            t = ts if isinstance(ts, dt.datetime) else ts.to_datetime()
        except Exception:
            try:
                t = dt.datetime.fromisoformat(str(ts))
            except Exception:
                continue
        if latest is None or t > latest:
            latest = t
    return latest

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="한국폴리텍대학 스마트금융과", layout="wide")
st.title("📊 종목리포트 평가 랭킹보드")

# ---- 면책 조항 ----
st.markdown("---")
st.markdown(
    """
    ⚠️ **면책 조항 (Disclaimer)**
    본 사이트는 **한국폴리텍대학 스마트금융과 학생들의 실습 목적**으로 제작된 것입니다.
    따라서 제공되는 데이터와 랭킹은 오류가 있을 수 있으며, 어떠한 공신력도 갖지 않습니다.
    또한, 본 사이트의 정보를 기반으로 한 **투자 결정 및 그 결과에 대한 책임은 전적으로 이용자 본인에게** 있습니다.
    제작자는 투자 손실 등 어떠한 법적 책임도 지지 않습니다.
    """,
    unsafe_allow_html=True
)

# ---- 사이드바 ----
with st.sidebar:
    st.header("Filters")
    today = dt.date.today()
    default_from = today - dt.timedelta(days=365)
    date_from = st.date_input("From date", value=default_from)
    date_to = st.date_input("To date", value=today)

    metric = st.radio("Ranking metric", ["avg", "sum"], index=0, horizontal=True)
    min_reports = st.slider("Minimum reports", 1, 20, 2, 1)
    max_docs = st.slider("Max docs to scan (by_analyst)", 500, 5000, 3000, 100)
    st.caption("※ 기간/브로커를 좁힐수록 빠릅니다. 점수는 리포트별 points_total 합계(또는 평균)를 사용합니다.")

    # 기간 기준으로 문서 로드 → 이용 가능한 브로커 집합/건수 계산
    base_docs = load_analyst_docs(date_from, date_to, brokers=None, limit=max_docs)
    broker_counts = {}
    for d in base_docs:
        b = normalize_broker_name((d.get("broker") or "").strip())
        if not b:
            continue
        broker_counts[b] = broker_counts.get(b, 0) + 1

    available_brokers = sorted([b for b, c in broker_counts.items() if c > 0])

    st.markdown("### 증권사 선택")

    if not available_brokers:
        st.info("해당 기간에 리포트가 존재하는 증권사가 없습니다. 기간을 조정해 주세요.")
        selected_brokers = []
    else:
        # 표시용 라벨: "증권사명 (리포트수)"
        display_labels = [f"{b} ({broker_counts[b]})" for b in available_brokers]
        # 기본 선택: 전체 선택 상태
        default_selection = display_labels

        # 멀티 셀렉트 UI
        selected_labels = st.multiselect(
            "리포트가 존재하는 증권사 목록",
            options=display_labels,
            default=default_selection,
            help="Ctrl/Cmd 클릭으로 다중 선택, 전체 선택 또는 해제 가능"
        )

        # 선택된 라벨에서 실제 broker 이름만 추출
        selected_brokers = [
            b.split(" (")[0] for b in selected_labels if " (" in b
        ]

        if not selected_brokers:
            st.warning("선택된 증권사가 없습니다. 최소 1개 이상 선택해 주세요.")


# ---- 데이터 로드 ----
with st.spinner("Loading analyst documents..."):
    docs = load_analyst_docs(date_from, date_to, selected_brokers or None, max_docs)

last_updated = get_last_updated_from_docs(docs)
st.markdown(f"최근 평가 반영 시각(문서 기준): {format_ts(last_updated)}")

# 평가 반영 기간 표시
try:
    _date_vals = [str(d.get("report_date", "")).strip() for d in docs]
    _date_vals = [s for s in _date_vals if s and len(s) >= 8]
    if _date_vals:
        _min_date = min(_date_vals)
        _max_date = max(_date_vals)
        st.markdown(f"평가 반영 기간 {_min_date} ~ {_max_date}")
    else:
        st.markdown("평가 반영 기간 - ~ -")
except Exception:
    st.markdown("평가 반영 기간 - ~ -")

# ---- 상단: Top Analysts (페이지네이션 그대로) ----
st.subheader("🏆 Top Analysts")

# 👇 랭킹 집계용으로 한 번 더 확실하게 날짜 필터 강제
docs_for_rank = filter_docs_by_date(docs, date_from, date_to)

# 디버그: 필터 전/후 건수 확인
st.caption(f"[DEBUG] ranking docs: before={len(docs)} after_date_filter={len(docs_for_rank)}")

rank_df = make_rank_table(docs_for_rank, metric=metric, min_reports=min_reports)


if rank_df.empty:
    st.info("조건에 맞는 데이터가 없습니다. 기간/브로커 필터를 조정하세요.")
    st.stop()

per_page = st.selectbox("Rows per page", [10,25,50,100], 1)
page = st.number_input("Page", 1, 9999, 1)
total_pages = max(1, math.ceil(len(rank_df)/per_page))
page = min(page, total_pages)
start, end = (page-1)*per_page, (page-1)*per_page + per_page
show_df = rank_df.iloc[start:end].copy()

st.caption(f"Page {page}/{total_pages} · Total analysts: {len(rank_df)}")

st.markdown("---")
st.subheader("🔍 하단 상세: 상단 표에서 행을 클릭하면 자동으로 표시됩니다")

# ---- 핵심: 행 선택 UI (Ag-Grid) ----
selected_idx = None
_show_df = show_df.reset_index(drop=True).copy()

# 행/헤더 높이 설정
ROW_H = 34
HEAD_H = 40
PADDING = 32
GRID_H = HEAD_H + ROW_H * len(_show_df) + PADDING   # 현재 페이지에 보여줄 행 수만큼
# 25행이 모두 보이도록 충분히 키우되, 과도한 높이 방지
GRID_H = min(GRID_H, HEAD_H + ROW_H * 25 + 400)      # 필요시 상단 상수 조절

if _AGGRID_AVAILABLE and not _show_df.empty:
    _show_df.insert(0, "_row", _show_df.index)

    gb = GridOptionsBuilder.from_dataframe(_show_df)
    # ✅ 단일 선택 (체크박스 없음)
    gb.configure_selection(selection_mode="single", use_checkbox=False)

    # ❌ autoHeight 제거 (내부 스크롤 사용)
    # gb.configure_grid_options(domLayout="autoHeight")
    gb.configure_grid_options(
        rowHeight=ROW_H,
        headerHeight=HEAD_H,
        # 필요시: suppressHorizontalScroll=True,
    )
    gb.configure_column("_row", header_name="", hide=True)

    grid = AgGrid(
        _show_df,
        gridOptions=gb.build(),
        update_mode=GridUpdateMode.SELECTION_CHANGED,
        data_return_mode=DataReturnMode.AS_INPUT,
        theme="streamlit",
        allow_unsafe_jscode=True,
        fit_columns_on_grid_load=True,
        height=GRID_H,     # 🔷 per_page에 맞춘 충분한 높이
    )

    selected_idx = _extract_selected_idx_from_aggrid(grid)

else:
    # ---- Fallback: st.dataframe + 드롭다운 선택 (동일 높이 적용) ----
    st.info("고급 행 클릭 선택을 위해 `streamlit-aggrid` 설치를 권장합니다. (fallback UI 사용 중)")
    df_to_show = _show_df.drop(columns=["_row"], errors="ignore") if "_row" in _show_df else _show_df
    st.dataframe(df_to_show, use_container_width=True, height=GRID_H)  # 🔷 동일 높이
    options = [f"{i}: {row['Analyst']} — {row['Broker']} (RankScore={row['RankScore']})"
               for i, row in _show_df.iterrows()]
    if options:
        pick_label = st.selectbox("상세를 볼 행 선택 (대체 UI)", options, index=0)
        try:
            selected_idx = int(pick_label.split(":")[0])
        except Exception:
            selected_idx = 0


# ---- 선택 없으면 안내 ----
if selected_idx is None:
    st.info("상단 표에서 행을 클릭(선택)하세요.")
    st.stop()

# ---- 선택된 애널리스트 정보 표시 + 상세 근거표 ----
picked = _show_df.iloc[selected_idx]
picked_name = picked.get("Analyst", "Unknown Analyst")
picked_broker = picked.get("Broker", "Unknown Broker")

st.markdown(f"### {picked_name} — {picked_broker}")
st.write(f"**Reports:** {picked.get('Reports', '-') } | **RankScore:** {picked.get('RankScore', '-') }")
st.write(f"**Coverage:** {picked.get('FirstReport', '-') } → {picked.get('LastReport', '-') }")

with st.spinner("Loading evidence..."):
    items = load_detail_for_analyst(picked_name, picked_broker, date_from, date_to)

if not items:
    st.info("상세 데이터가 없습니다.")
    st.stop()

# 리포트별 Horizon 근거를 길게 펼쳐서 한 테이블로 구성
evidence_rows: List[Dict[str, Any]] = []
for it in items:
    evidence_rows.extend(horizon_to_rows(it))

if not evidence_rows:
    st.warning("Horizon 근거가 비어 있습니다.")
else:
    ev_df = pd.DataFrame(evidence_rows)

    # 보기 좋게 컬럼 순서 재배치
    wanted_cols = [
        "보고서일", "종목", "티커", "레이팅", "예측포지션", "예측목표가",
        "horizon(일)",
        "구간최고종가", "구간최저종가", "구간마지막종가", "목표가대비최대도달%",
        "리포트일에 매수했을경우 수익률%", "방향정답", "방향점수", "목표근접점수", "목표가HIT",
        "리포트총점", "제목", "리포트PDF", "상세페이지"
    ]
    cols = [c for c in wanted_cols if c in ev_df.columns] + [c for c in ev_df.columns if c not in wanted_cols]
    ev_df = ev_df[cols]

    # 링크 컬럼 정리 (스킴 누락 대비)
    if "리포트PDF" in ev_df.columns:
        ev_df["리포트PDF"] = ev_df["리포트PDF"].map(_norm_url)
    if "상세페이지" in ev_df.columns:
        ev_df["상세페이지"] = ev_df["상세페이지"].map(_norm_url)

    # Streamlit 최신 버전: LinkColumn 사용
    try:
        st.dataframe(
            ev_df,
            use_container_width=True,
            column_config={
                "리포트PDF": st.column_config.LinkColumn(
                    label="리포트PDF",
                    help="PDF 열기",
                    display_text="열기"
                ),
                "상세페이지": st.column_config.LinkColumn(
                    label="상세페이지",
                    help="상세 페이지 열기",
                    display_text="열기"
                ),
            },
        )
    except Exception:
        # 구버전 폴백: HTML 테이블
        def _a(href):
            return f'<a href="{href}" target="_blank" rel="noopener noreferrer">열기</a>' if href else ""
        html_df = ev_df.copy()
        if "리포트PDF" in html_df.columns:
            html_df["리포트PDF"] = html_df["리포트PDF"].map(_a)
        if "상세페이지" in html_df.columns:
            html_df["상세페이지"] = html_df["상세페이지"].map(_a)

        st.markdown(html_df.to_html(escape=False, index=False), unsafe_allow_html=True)

    st.caption("※ 한국폴리텍대학 스마트금융과 실습 목적의 결과물로 데이터 취합, 정제, 분석이 부정확 할 수 있음.")
