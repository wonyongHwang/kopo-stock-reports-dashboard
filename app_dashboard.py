# -*- coding: utf-8 -*-
"""
Analyst Ranking & Evidence Dashboard (Horizon-only)
- 상위 랭커를 바로 보여주고, 클릭 시 근거(리포트 URL/예측가/포지션/실제 성과)를 테이블로 노출
- 30/60일 성과 섹션은 제거 (요청사항)
- Firestore 구조는 evaluations/by_analyst 저장 스키마에 맞춤
"""

import math
import datetime as dt
from typing import List, Dict, Any, Tuple

import streamlit as st
import pandas as pd
import datetime as dt
import pytz

from google.cloud import firestore

# app_dashboard.py 상단에 전역 설정 추가
EXCLUDED_BROKERS = {"한국IR협의회"}  # 제외할 증권사명 집합

def is_allowed_broker(broker: str) -> bool:
    return broker not in EXCLUDED_BROKERS

# 파일 상단 헬퍼들 근처에 추가
def _norm_url(x: str) -> str:
    if not x:
        return ""
    s = str(x).strip()
    if s.startswith(("http://", "https://")):
        return s
    # 스킴이 없으면 기본 https 가정
    return "https://" + s



KST = pytz.timezone("Asia/Seoul")

def format_ts(ts, tz=KST):
    """Firestore Timestamp 또는 문자열/None → 보기용 문자열"""
    if not ts:
        return "-"
    try:
        # Firestore Timestamp -> datetime
        dt_utc = ts if isinstance(ts, dt.datetime) else ts.to_datetime()
    except Exception:
        # '2025-09-05' 같은 문자열일 수도 있음
        try:
            dt_utc = dt.datetime.fromisoformat(str(ts))
        except Exception:
            return str(ts)
    if dt_utc.tzinfo is None:
        dt_utc = dt_utc.replace(tzinfo=dt.timezone.utc)
    return dt_utc.astimezone(tz).strftime("%Y-%m-%d %H:%M:%S %Z")
# ===============================
# Firestore Client & Caches
# ===============================
@st.cache_resource(show_spinner=False)
def get_db():
    return firestore.Client()

@st.cache_data(show_spinner=False, ttl=60)
def load_broker_list() -> List[str]:
    """by_analyst 그룹에서 broker 목록 추출"""
    db = get_db()
    q = db.collection_group("by_analyst").select(["broker"]).limit(3000)
    s = set()
    try:
        for d in q.stream():
            b = (d.to_dict() or {}).get("broker")
            if b:
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
    broker_set = set(brokers or [])
    for d in docs:
        x = d.to_dict() or {}
        broker = (x.get("broker") or "").strip()
        if not is_allowed_broker(broker):
            continue   # ← 제외
        if broker_set and (x.get("broker") or "") not in broker_set:
            continue
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
    """
    db = get_db()
    out: List[Dict[str, Any]] = []
    q = (db.collection_group("by_analyst")
           .where("broker", "==", broker)
           .where("analyst_name", "==", analyst_name))
    try:
        if date_from: q = q.where("report_date", ">=", date_from.isoformat())
        if date_to:   q = q.where("report_date", "<=", date_to.isoformat())
    except Exception:
        pass
    q = q.limit(limit)
    for d in q.stream():
        out.append(d.to_dict() or {})
    out.sort(key=lambda r: r.get("report_date",""), reverse=True)
    return out

# ===============================
# Helpers (표현/집계)
# ===============================
# (상단 helpers 근처에 추가)
def _price_stats(price_sample: list[dict]) -> tuple[float|None, float|None, float|None]:
    """
    price_sample: [{"date": "...", "open":.., "high":.., "low":.., "close":..}, ...]
    return: (max_close, min_close, last_close)
    """
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

# 기존 horizon_to_rows 를 아래로 교체
def horizon_to_rows(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    by_analyst 한 문서(= 한 리포트) → 여러 horizon 근거 행으로 펼침
    - 기존 return_pct (= start_close 대비 end_close 변화율)
    - 추가: 목표가 대비 수익률% (= (end_close - target_price)/target_price )
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

            # 신규: 목표가 대비
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



# def make_rank_table(docs: List[Dict[str, Any]],
#                     metric: str = "avg", min_reports: int = 2) -> pd.DataFrame:
#     """
#     애널리스트별 집계 → 랭킹표 생성
#     metric: "avg" | "sum"
#     """
#     from collections import defaultdict
#     agg = defaultdict(lambda: {"sum":0.0, "n":0, "name":"", "broker":"", "first":None, "last":None})
#     for x in docs:
#         name = normalize_name(x.get("analyst_name","")) or "(unknown)"
#         broker = x.get("broker","")
#         pts = float(x.get("points_total") or 0.0)
#         rdate = x.get("report_date","")

#         key = f"{name}|{broker}"
#         agg[key]["sum"] += pts
#         agg[key]["n"] += 1
#         agg[key]["name"] = name
#         agg[key]["broker"] = broker
#         if rdate:
#             if not agg[key]["first"] or rdate < agg[key]["first"]:
#                 agg[key]["first"] = rdate
#             if not agg[key]["last"]  or rdate > agg[key]["last"]:
#                 agg[key]["last"]  = rdate

#     rows = []
#     for k,v in agg.items():
#         if v["n"] < min_reports:  # 최소 리포트 수
#             continue
#         score = (v["sum"]/v["n"]) if metric=="avg" else v["sum"]
#         rows.append({
#             "RankScore": round(score,4),
#             "Reports": v["n"],
#             "Analyst": v["name"],
#             "Broker": v["broker"],
#             "FirstReport": v["first"] or "",
#             "LastReport": v["last"] or "",
#         })
#     df = pd.DataFrame(rows).sort_values(["RankScore","Reports"], ascending=[False, False]).reset_index(drop=True)
#     df.index = df.index + 1  # 1-based rank
#     return df

def make_rank_table(docs: list[dict], metric: str = "avg", min_reports: int = 2) -> pd.DataFrame:
    from collections import defaultdict
    agg = defaultdict(lambda: {"sum":0.0, "n":0, "name":"", "broker":"", "first":None, "last":None})

    # 0) 입력이 비면 즉시 빈 DF 반환
    if not docs:
        return pd.DataFrame(columns=["RankScore","Reports","Analyst","Broker","FirstReport","LastReport"])

    # 1) 집계
    for x in docs:
        name = (x.get("analyst_name") or "").strip() or "(unknown)"
        broker = (x.get("broker") or "").strip()
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

    # 2) 순위행 생성
    rows = []
    for k, v in agg.items():
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

    # 3) rows가 비면 안전하게 빈 DF 반환 (정렬 시 KeyError 방지)
    if not rows:
        return pd.DataFrame(columns=["RankScore","Reports","Analyst","Broker","FirstReport","LastReport"])

    df = pd.DataFrame(rows)
    # 4) 혹시라도 컬럼 누락 시 방어
    if not {"RankScore","Reports"}.issubset(df.columns):
        return df.reset_index(drop=True)

    return df.sort_values(["RankScore","Reports"], ascending=[False, False]).reset_index(drop=True)

def get_last_updated_from_docs(docs):
    latest = None
    for d in docs:
        ts = d.get("updated_at")
        if ts is None:
            continue
        # Firestore Timestamp 비교
        try:
            t = ts if isinstance(ts, dt.datetime) else ts.to_datetime()
        except Exception:
            # 문자열일 수도 있음
            try:
                t = dt.datetime.fromisoformat(str(ts))
            except Exception:
                continue
        if latest is None or t > latest:
            latest = t
    return latest

# ===============================
# Streamlit UI
# ===============================
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

# Sidebar Filters
# with st.sidebar:
#     st.header("Filters")
#     today = dt.date.today()
#     default_from = today - dt.timedelta(days=365)
#     date_from = st.date_input("From date", value=default_from)
#     date_to = st.date_input("To date", value=today)
#     brokers = load_broker_list()
#     sel_brokers = st.multiselect("Brokers", brokers, default=[])
#     metric = st.radio("Ranking metric", ["avg", "sum"], index=0, horizontal=True)
#     min_reports = st.slider("Minimum reports", 1, 20, 2, 1)
#     max_docs = st.slider("Max docs to scan (by_analyst)", 500, 5000, 3000, 100)
#     st.caption("※ 기간/브로커를 좁힐수록 빠릅니다. 점수는 리포트별 points_total 합계(또는 평균)를 사용합니다.")
# ---- 기존 사이드바 일부 교체 ----
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

    # ➊ 먼저 기간 기준으로만 문서 로드 → 이용 가능한 브로커 집합/건수 계산
    base_docs = load_analyst_docs(date_from, date_to, brokers=None, limit=max_docs)
    broker_counts = {}
    for d in base_docs:
        b = (d.get("broker") or "").strip()
        if not b:
            continue
        broker_counts[b] = broker_counts.get(b, 0) + 1

    # 이용 가능한 브로커만 (건수>0) 정렬
    available_brokers = sorted([b for b, c in broker_counts.items() if c > 0])

    st.markdown("### 증권사 선택 (괄호안은 리포트 건수)")
    if not available_brokers:
        st.info("해당 기간에 리포트가 존재하는 증권사가 없습니다. 기간을 조정해 주세요.")
        selected_brokers = []
    else:
        # 전체 선택 토글
        select_all = st.checkbox("전체 선택", value=True, key="brokers_select_all")

        # ➋ 체크박스 렌더: (이 기간에 리포트 있는 브로커만) 라벨에 건수 표기
        selected_brokers = []
        for b in available_brokers:
            label = f"{b} ({broker_counts.get(b, 0)})"
            checked = st.checkbox(label, value=select_all, key=f"broker_{b}")
            if checked:
                selected_brokers.append(b)

        # 아무 것도 선택 안되면 경고 (집계는 빈 결과가 되므로)
        if not selected_brokers:
            st.warning("선택된 증권사가 없습니다. 최소 1개 이상 선택해 주세요.")

# Load
with st.spinner("Loading analyst documents..."):
    # docs = load_analyst_docs(date_from, date_to, sel_brokers or None, max_docs)
    docs = load_analyst_docs(date_from, date_to, selected_brokers or None, max_docs)
    # st.write(f"[DEBUG] loaded={len(docs)}, broker_filter={sel_brokers or '(all)'}")

last_updated = get_last_updated_from_docs(docs)
st.markdown(f"최근 평가 반영 시각(문서 기준): {format_ts(last_updated)}")

# 추가된 코드
# --- 평가 반영 기간(리포트 날짜 범위) 표시 ---
try:
    # docs에서 report_date 문자열만 추출하여 유효한 것들만 필터
    _date_vals = [str(d.get("report_date", "")).strip() for d in docs]
    _date_vals = [s for s in _date_vals if s and len(s) >= 8]  # 대략 'YYYY-MM-DD' 형태
    if _date_vals:
        _min_date = min(_date_vals)
        _max_date = max(_date_vals)
        st.markdown(f"평가 반영 기간 {_min_date} ~ {_max_date}")
    else:
        st.markdown("평가 반영 기간 - ~ -")
except Exception:
    st.markdown("평가 반영 기간 - ~ -")
# --- End ---

# Rank table (shown immediately)
st.subheader("🏆 Top Analysts")
rank_df = make_rank_table(docs, metric=metric, min_reports=min_reports)
st.write(f"[DEBUG] after min_reports=1 -> candidates={len(rank_df)} (현재 설정 min_reports={min_reports})")

if rank_df.empty:
    st.info("조건에 맞는 데이터가 없습니다. 기간/브로커 필터를 조정하세요.")
    st.stop()

# 페이지네이션 (간단)
per_page = st.selectbox("Rows per page", [10,25,50,100], 1)
page = st.number_input("Page", 1, 9999, 1)
total_pages = max(1, math.ceil(len(rank_df)/per_page))
page = min(page, total_pages)
start, end = (page-1)*per_page, (page-1)*per_page + per_page
show_df = rank_df.iloc[start:end].copy()

st.dataframe(show_df, use_container_width=True)
st.caption(f"Page {page}/{total_pages} · Total analysts: {len(rank_df)}")

# Pick one analyst
st.markdown("---")
st.subheader("🔍 애널리스트 클릭시 화면 하단에 상세 평가표 조회 가능")

# 3열 버튼 UI (이름 + 점수), 클릭 시 아래에 상세 표를 인라인으로 표시
if "picked_row_idx" not in st.session_state:
    st.session_state.picked_row_idx = None

_tmp_df_btn = show_df.reset_index(drop=True).copy()
n_cols = 3
cols = st.columns(n_cols)
for i, row in _tmp_df_btn.iterrows():
    with cols[i % n_cols]:
        btn_label = f"{row.get('Analyst', 'Unknown')} · {row.get('RankScore', '-')}"
        if st.button(btn_label, key=f"pick_{page}_{i}", use_container_width=True):
            st.session_state.picked_row_idx = i

# 선택이 없으면 중단
if st.session_state.picked_row_idx is None:
    st.stop()

# 선택된 애널리스트 파싱
picked = _tmp_df_btn.iloc[st.session_state.picked_row_idx]
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
    # 근거표를 렌더하는 부분에서 ev_df 생성 후 정렬 컬럼 지정하는 리스트 갱신
    wanted_cols = [
    "보고서일", "종목", "티커", "레이팅", "예측포지션", "예측목표가",
    "horizon(일)",

    # 비교용 신규 지표
    "구간최고종가", "구간최저종가", "구간마지막종가", "목표가대비최대도달%",

    "리포트일에 매수했을경우 수익률%", "방향정답", "방향점수", "목표근접점수", "목표가HIT",
    "리포트총점", "제목", "리포트PDF", "상세페이지"]
    cols = [c for c in wanted_cols if c in ev_df.columns] + [c for c in ev_df.columns if c not in wanted_cols]
    ev_df = ev_df[cols]

    # 링크 컬럼 정리 (스킴 누락 대비)
    if "리포트PDF" in ev_df.columns:
        ev_df["리포트PDF"] = ev_df["리포트PDF"].map(_norm_url)
    if "상세페이지" in ev_df.columns:
        ev_df["상세페이지"] = ev_df["상세페이지"].map(_norm_url)

    # Streamlit 1.23+ : LinkColumn 사용 (권장)
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
        # 구버전 Streamlit 대응: HTML 테이블로 폴백
        def _a(href):
            return f'<a href="{href}" target="_blank" rel="noopener noreferrer">열기</a>' if href else ""
        html_df = ev_df.copy()
        if "리포트PDF" in html_df.columns:
            html_df["리포트PDF"] = html_df["리포트PDF"].map(_a)
        if "상세페이지" in html_df.columns:
            html_df["상세페이지"] = html_df["상세페이지"].map(_a)

        st.markdown(
            html_df.to_html(escape=False, index=False),
            unsafe_allow_html=True
        )

    # st.caption("※ 링크 컬럼을 클릭하면 새 탭으로 열립니다.")


    # 링크 컬럼은 클릭 가능하게 렌더
    # st.dataframe(ev_df, use_container_width=True)
    st.caption("※ 한국폴리텍대학 스마트금융과 실습 목적의 결과물로 데이터 취합, 정제, 분석이 부정확 할 수 있음.")
