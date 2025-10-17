# -*- coding: utf-8 -*-
"""
종목리포트 평가 대시보드 (완성본; 기존 기능 유지)
- 탭1: 🏆 종목리포트 평가 랭킹보드 (행 선택 → 하단 상세 근거 표시, 페이지네이션 포함)
- 탭2: 🔎 종목별 검색 (종목/티커로 조회, 증권사/애널리스트 요약 + 상세)
- 면책 조항 / 평가 반영 기간 & 최근 반영 시각 표기
- 브로커/애널리스트 공백 제거 정규화
- Ag-Grid 선택값이 list/DF 모두에서 안전하게 동작
"""

import math
import datetime as dt
from typing import List, Dict, Any
import re

import streamlit as st
import pandas as pd
import pytz
from google.cloud import firestore

# ====== Ag-Grid 옵션 ======
_AGGRID_AVAILABLE = True
try:
    from st_aggrid import AgGrid, GridOptionsBuilder, GridUpdateMode, DataReturnMode
except Exception:
    _AGGRID_AVAILABLE = False

# -----------------------------
# 정규화 & 유틸
# -----------------------------
def normalize_broker_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r"\s+", "", name)           # 모든 공백 제거
    name = re.sub(r"[&/\.#\[\]]", "_", name)  # Firestore 예약문자 치환
    name = re.sub(r"[\(\)\[\]]", "", name)    # 괄호류 제거
    return name.strip()

def normalize_analyst_name(name: str) -> str:
    if not name:
        return ""
    return re.sub(r"\s+", "", str(name)).strip()

def _norm_url(x: str) -> str:
    if not x:
        return ""
    s = str(x).strip()
    if s.startswith(("http://", "https://")):
        return s
    return "https://" + s

KST = pytz.timezone("Asia/Seoul")

def format_ts(ts, tz=KST):
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

def _aggrid_selected_empty(sel) -> bool:
    """AgGrid selected_rows 가 list 또는 DataFrame 양쪽을 안전하게 비었는지 판별"""
    if isinstance(sel, list):
        return len(sel) == 0
    if isinstance(sel, pd.DataFrame):
        return sel.empty
    return True

def _aggrid_pick_first(sel):
    """AgGrid selected_rows 에서 첫 행 dict로 꺼내기 (list/DF 모두 지원)"""
    if isinstance(sel, list):
        return sel[0] if sel else None
    if isinstance(sel, pd.DataFrame):
        return sel.iloc[0].to_dict() if not sel.empty else None
    return None

# --- [추가] horizons에서 가장 최근(최신 end_date) 구간의 마지막 종가 추출 ---
def last_close_from_horizons(rec: dict) -> float | None:
    best_close = None
    best_key = None
    for h in (rec.get("horizons") or []):
        sample = h.get("price_sample") or []
        if not sample:
            continue
        # sample[-1]이 가장 최근 시점
        last = sample[-1]
        close = last.get("close")
        # 비교 기준: end_date(없으면 sample의 마지막 date) 문자열 비교
        end_key = (h.get("end_date") or last.get("date") or "")
        if close is None:
            continue
        if best_key is None or str(end_key) > str(best_key):
            best_key = end_key
            try:
                best_close = float(close)
            except Exception:
                pass
    return best_close

# -----------------------------
# Firestore Client
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_db():
    return firestore.Client()

# -----------------------------
# 데이터 로드
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def load_analyst_docs(date_from: dt.date, date_to: dt.date,
                      brokers: List[str] | None, limit: int = 3000) -> List[Dict[str, Any]]:
    """evaluations/by_analyst 그룹에서 기간/브로커 필터로 문서 로드"""
    db = get_db()
    want_min = date_from.isoformat() if date_from else None
    want_max = date_to.isoformat() if date_to else None

    docs_raw = []
    server_filtered = False
    try:
        q = db.collection_group("by_analyst")
        if want_min: q = q.where("report_date", ">=", want_min)
        if want_max: q = q.where("report_date", "<=", want_max)
        q = q.limit(limit)
        docs_raw = list(q.stream())
        server_filtered = True
    except Exception:
        q = db.collection_group("by_analyst").limit(limit)
        docs_raw = list(q.stream())

    broker_set = {normalize_broker_name(b) for b in (brokers or [])} or None
    out: List[Dict[str, Any]] = []
    for d in docs_raw:
        x = d.to_dict() or {}
        broker_norm = normalize_broker_name((x.get("broker") or "").strip())
        # ✅ '한국IR협의회'는 모든 조회에서 제외
        if broker_norm == "한국IR협의회" or "평가" in broker_norm :
            continue
        if broker_set and broker_norm not in broker_set:
            continue
        x["broker"] = broker_norm
        # 애널리스트 이름 정규화 보조 필드
        raw_name = (x.get("analyst_name") or x.get("analyst") or "").strip()
        x["analyst_name"] = raw_name
        x["analyst_name_norm"] = x.get("analyst_name_norm") or normalize_analyst_name(raw_name)

        # 클라이언트 쪽 기간 필터 fallback
        if not server_filtered and (date_from or date_to):
            s = (x.get("report_date") or "").strip()
            if not s:
                continue
            try:
                d0 = dt.date.fromisoformat(s[:10])
            except Exception:
                continue
            if date_from and d0 < date_from:
                continue
            if date_to and d0 > date_to:
                continue

        # points_total 보정
        if x.get("points_total") is None and x.get("horizons"):
            x["points_total"] = sum(float(h.get("points_total_this_horizon") or 0.0) for h in x.get("horizons") or [])
        out.append(x)

    return out

@st.cache_data(show_spinner=False, ttl=60)
def load_detail_for_analyst(analyst_name: str, broker: str,
                            date_from: dt.date | None, date_to: dt.date | None,
                            limit: int = 800) -> List[Dict[str, Any]]:
    """특정 애널리스트(이름+브로커)의 근거 문서들"""
    db = get_db()
    out: List[Dict[str, Any]] = []
    broker_norm = normalize_broker_name(broker)
    target_norm = normalize_analyst_name(analyst_name)

    def _q(field: str, value: str):
        q = (db.collection_group("by_analyst")
               .where("broker", "==", broker_norm)
               .where(field, "==", value))
        try:
            if date_from: q = q.where("report_date", ">=", date_from.isoformat())
            if date_to:   q = q.where("report_date", "<=", date_to.isoformat())
        except Exception:
            pass
        q = q.limit(limit)
        docs = [z.to_dict() or {} for z in q.stream()]
        for z in docs:
            z["broker"] = normalize_broker_name(z.get("broker"))
            rn = (z.get("analyst_name") or z.get("analyst") or "").strip()
            z["analyst_name"] = rn
            z["analyst_name_norm"] = z.get("analyst_name_norm") or normalize_analyst_name(rn)
        return docs

    out.extend(_q("analyst_name_norm", target_norm))
    if not out:
        out.extend(_q("analyst_name", analyst_name))
        if target_norm != analyst_name:
            out.extend(_q("analyst_name", target_norm))

    # 중복 제거
    seen = set(); uniq = []
    for r in out:
        key = (str(r.get("report_id","")), str(r.get("pdf_url","")))
        if key in seen:
            continue
        seen.add(key); uniq.append(r)
    uniq.sort(key=lambda r: r.get("report_date",""), reverse=True)
    return uniq

# -----------------------------
# 집계 & 테이블 변환
# -----------------------------
def make_rank_table(docs: list[dict], metric: str = "avg", min_reports: int = 2) -> pd.DataFrame:
    from collections import defaultdict
    agg = defaultdict(lambda: {"sum":0.0, "n":0, "name_disp":"", "broker":"", "first":None, "last":None})
    for x in docs:
        name_raw = (x.get("analyst_name") or x.get("analyst") or "").strip()
        name_norm = x.get("analyst_name_norm") or normalize_analyst_name(name_raw)
        broker = normalize_broker_name((x.get("broker") or "").strip())
        pts = 0.0
        try: pts = float(x.get("points_total") or 0.0)
        except Exception: pass
        rdate = (x.get("report_date") or "").strip()
        key = f"{name_norm}|{broker}"
        agg[key]["sum"] += pts
        agg[key]["n"] += 1
        if not agg[key]["name_disp"]:
            agg[key]["name_disp"] = name_raw or "(unknown)"
        agg[key]["broker"] = broker
        if rdate:
            if not agg[key]["first"] or rdate < agg[key]["first"]:
                agg[key]["first"] = rdate
            if not agg[key]["last"] or rdate > agg[key]["last"]:
                agg[key]["last"]  = rdate
    rows = []
    for _, v in agg.items():
        if v["n"] < min_reports: continue
        score = (v["sum"]/v["n"]) if metric == "avg" else v["sum"]
        rows.append({
            "RankScore": round(score, 4),
            "Reports": v["n"],
            "Analyst": v["name_disp"],
            "Broker": v["broker"],
            "FirstReport": v["first"] or "",
            "LastReport": v["last"] or "",
        })
    if not rows:
        return pd.DataFrame(columns=["RankScore","Reports","Analyst","Broker","FirstReport","LastReport"])
    df = pd.DataFrame(rows).sort_values(["RankScore","Reports"], ascending=[False, False]).reset_index(drop=True)
    return df

def _price_stats(price_sample: list[dict] | None):
    if not price_sample:
        return None, None, None
    closes = [float(p.get("close")) for p in price_sample if p.get("close") is not None]
    if not closes:
        return None, None, None
    return max(closes), min(closes), closes[-1]

def _sign(x: float | None):
    if x is None: return None
    return 1 if x > 0 else (-1 if x < 0 else 0)

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
            if desired == 0: correct = (actual_sign == 0)
            else:            correct = (desired == actual_sign)

        sample = h.get("price_sample") or []
        max_c, min_c, last_c = _price_stats(sample)

        # 목표가 대비 수익률 (마지막 종가 기준)
        target_return_pct = None
        if target_price and target_price not in ("", 0, 0.0) and last_c:
            try:
                target_return_pct = round((float(last_c)-float(target_price))/float(target_price)*100, 3)
            except Exception:
                target_return_pct = None

        # 목표가 대비 최대 도달률 (최고가 기준)
        max_reach_pct = None
        if target_price and target_price not in ("", 0, 0.0) and max_c:
            try:
                max_reach_pct = round((float(max_c)/float(target_price)-1)*100, 3)
            except Exception:
                max_reach_pct = None

        rows.append({
            "보고서일": rec.get("report_date",""),
            "horizon(일)": h.get("horizon_days"),
            "예측포지션": format_pos_label(desired, rating_norm),
            "레이팅": rating_norm,
            "예측목표가": target_price,
            "리포트일에 매수했을경우 수익률%": None if ret is None else round(ret*100, 3),
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

def match_stock(rec: dict, stock_q: str | None, ticker_q: str | None) -> bool:
    if not stock_q and not ticker_q:
        return True
    if ticker_q:
        t = (rec.get("ticker") or "").strip()
        if t and str(t).lower() == str(ticker_q).lower():
            return True
    if stock_q:
        name = (rec.get("stock") or "").strip()
        if name and str(stock_q).lower() in name.lower():
            return True
    return False

def get_last_updated_from_docs(docs):
    latest = None
    for d in docs:
        ts = d.get("updated_at")
        if not ts:
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
st.title("📊 종목리포트 분석")
# --- Tabs 시인성 향상 (CSS 스타일 주입: 최소 변경) ---
st.markdown("""
<style>
/* 탭 묶음 하단 경계선 & 간격 */
div[data-testid="stTabs"] div[role="tablist"] {
  border-bottom: 2px solid #e9ecef;
  gap: 8px;
  padding-bottom: 2px;
  margin-bottom: 8px;
}

/* 각 탭 버튼: 기본(비활성) 스타일 */
div[data-testid="stTabs"] button[role="tab"] {
  font-weight: 600;
  font-size: 0.95rem;
  color: #495057;
  background: #f8f9fa;
  border: 1px solid #e9ecef;
  border-bottom: none; /* 아래쪽은 콘텐츠 카드와 연결되므로 없앰 */
  border-top-left-radius: 8px;
  border-top-right-radius: 8px;
  padding: 10px 14px;
  box-shadow: inset 0 -3px 0 0 rgba(0,0,0,0.03);
}

/* Hover 시 살짝 강조 */
div[data-testid="stTabs"] button[role="tab"]:hover {
  background: #f1f3f5;
  color: #343a40;
}

/* 활성 탭 */
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  background: #ffffff;
  color: #212529;
  border-color: #dee2e6;
  box-shadow: inset 0 -4px 0 0 #0d6efd; /* 상단 파란 강조선(브랜드 컬러 느낌) */
}

/* 탭 패널(내용) 카드 스타일 */
div[data-testid="stTabs"] > div[data-baseweb="tab-panel"] {
  border: 1px solid #dee2e6;
  border-top: none; /* 활성 탭 버튼과 자연스럽게 이어지도록 */
  border-radius: 0 10px 10px 10px;
  padding: 16px 16px 10px 16px;
  background: #ffffff;
}

/* 작은 화면에서 탭 라벨이 두 줄이 될 때 줄 간격 */
div[data-testid="stTabs"] button[role="tab"] p {
  line-height: 1.1;
  margin: 0;
}
</style>
""", unsafe_allow_html=True)

# ---- 사이드바 ----
with st.sidebar:
    st.header("공통 필터")
    today = dt.date.today()
    default_from = today - dt.timedelta(days=90)
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
        labels = [f"{b} ({broker_counts[b]})" for b in available_brokers]
        selected_labels = st.multiselect("리포트가 존재하는 증권사 목록", options=labels, default=labels, key="multi_brokers")
        selected_brokers = [lab.split(" (")[0] for lab in selected_labels if " (" in lab]
        if not selected_brokers:
            st.warning("선택된 증권사가 없습니다. 최소 1개 이상 선택해 주세요.")

# ---- 데이터 로드 (두 탭에서 공통 사용) ----
with st.spinner("문서 로딩 중..."):
    docs = load_analyst_docs(date_from, date_to, selected_brokers or None, max_docs)

# ⚠️ 면책 조항
st.markdown("---")
st.markdown(
    """
    ⚠️ **면책 조항 (Disclaimer)**
    본 사이트는 **한국폴리텍대학 스마트금융과 학생들의 실습 목적**으로 제작된 것입니다.
    따라서 제공되는 데이터와 랭킹은 오류가 있을 수 있으며, 어떠한 공신력도 갖지 않습니다.
    또한, 본 사이트의 정보를 기반으로 한 **투자 결정 및 그 결과에 대한 책임은 전적으로 이용자 본인**에게 있습니다.
    제작자는 투자 손실 등 어떠한 법적 책임도 지지 않습니다.
    """,
    unsafe_allow_html=True
)

# 평가 반영 기간 및 최근 갱신 시각
date_values = [str(d.get("report_date", "")).strip() for d in docs if str(d.get("report_date","")).strip()]
if date_values:
    try:
        min_date = min(date_values)
        max_date = max(date_values)
        st.markdown(f"**평가 반영 기간** {min_date} ~ {max_date}")
    except Exception:
        st.markdown("**평가 반영 기간** - ~ -")
else:
    st.markdown("**평가 반영 기간** - ~ -")

st.caption(f"최근 평가 반영 시각(문서 기준): {format_ts(get_last_updated_from_docs(docs))}")

# -----------------------------
# Tabs
# -----------------------------
tab_rank, tab_stock = st.tabs(["🏆 종목리포트 평가 랭킹보드", "🔎 종목별 검색"])

# ===============================
# 탭1: 랭킹보드 (페이지네이션 복원)
# ===============================
with tab_rank:
    st.subheader("🏆 Top Analysts")
    rank_df = make_rank_table(docs, metric=metric, min_reports=min_reports)

    if rank_df.empty:
        st.info("조건에 맞는 데이터가 없습니다. 기간/브로커/최소 리포트 수를 조정하세요.")
    else:
        # --- 페이지네이션(복원) ---
        per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1, key="rank_rows_per_page")
        page = st.number_input("Page", 1, 9999, 1, key="rank_page")
        total_pages = max(1, math.ceil(len(rank_df) / per_page))
        page = min(page, total_pages)
        start, end = (page - 1) * per_page, (page - 1) * per_page + per_page
        show_df = rank_df.iloc[start:end].copy()
        st.caption(f"Page {page}/{total_pages} · Total analysts: {len(rank_df)}")

        # Ag-Grid 렌더링 (선택 안정화)
        selected_idx = None
        _show_df = show_df.reset_index(drop=True).copy()
        ROW_H, HEAD_H, PADDING = 34, 40, 32
        GRID_H = min(HEAD_H + ROW_H * len(_show_df) + PADDING, HEAD_H + ROW_H * 25 + 400)

        if _AGGRID_AVAILABLE and not _show_df.empty:
            _show_df.insert(0, "_row", _show_df.index)
            gb = GridOptionsBuilder.from_dataframe(_show_df)
            gb.configure_selection(selection_mode="single", use_checkbox=False)
            gb.configure_grid_options(rowHeight=ROW_H, headerHeight=HEAD_H)
            gb.configure_column("_row", header_name="", hide=True)
            grid = AgGrid(
                _show_df,
                gridOptions=gb.build(),
                update_mode=GridUpdateMode.SELECTION_CHANGED,
                data_return_mode=DataReturnMode.AS_INPUT,
                theme="streamlit",
                allow_unsafe_jscode=True,
                fit_columns_on_grid_load=True,
                height=GRID_H,
            )
            sel = grid.get("selected_rows", [])
            if _aggrid_selected_empty(sel):
                st.info("상단 표에서 행을 클릭(선택)하세요.")
            else:
                picked = _aggrid_pick_first(sel)
                if not picked:
                    st.info("선택 행을 해석할 수 없습니다. 다시 선택해 주세요.")
                else:
                    picked_name = picked.get("Analyst", "Unknown Analyst")
                    picked_broker = picked.get("Broker", "Unknown Broker")
                    st.markdown(f"### {picked_name} — {picked_broker}")
                    st.write(f"**Reports:** {picked.get('Reports', '-') } | **RankScore:** {picked.get('RankScore', '-') }")
                    # 커버리지(선택된 페이징 내 first/last 대신 전체 테이블 기준 표기 필요 시 rank_df에서 찾을 수도 있음)
                    # 간단히 현재 슬라이스 값으로 표기
                    st.write(f"**Coverage:** {picked.get('FirstReport', '-') } → {picked.get('LastReport', '-') }")

                    with st.spinner("Loading evidence..."):
                        items = load_detail_for_analyst(picked_name, picked_broker, date_from, date_to)

                    if not items:
                        st.info("상세 데이터가 없습니다.")
                    else:
                        evidence_rows: List[Dict[str, Any]] = []
                        for it in items:
                            evidence_rows.extend(horizon_to_rows(it))
                        ev_df = pd.DataFrame(evidence_rows)

                        # 컬럼 순서
                        wanted = [
                            "보고서일", "종목", "티커", "레이팅", "예측포지션", "예측목표가",
                            "horizon(일)",
                            "구간최고종가", "구간최저종가", "구간마지막종가", "목표가대비최대도달%",
                            "리포트일에 매수했을경우 수익률%", "방향정답", "방향점수", "목표근접점수", "목표가HIT",
                            "리포트총점", "제목", "리포트PDF", "상세페이지"
                        ]
                        cols = [c for c in wanted if c in ev_df.columns] + [c for c in ev_df.columns if c not in wanted]
                        ev_df = ev_df[cols]

                        # 링크 맵
                        if "리포트PDF" in ev_df.columns:
                            ev_df["리포트PDF"] = ev_df["리포트PDF"].map(_norm_url)
                        if "상세페이지" in ev_df.columns:
                            ev_df["상세페이지"] = ev_df["상세페이지"].map(_norm_url)

                        try:
                            st.dataframe(
                                ev_df,
                                use_container_width=True,
                                column_config={
                                    "리포트PDF": st.column_config.LinkColumn(label="리포트PDF", display_text="열기"),
                                    "상세페이지": st.column_config.LinkColumn(label="상세페이지", display_text="열기"),
                                },
                            )
                        except Exception:
                            def _a(href):
                                return f'<a href="{href}" target="_blank" rel="noopener noreferrer">열기</a>' if href else ""
                            html_df = ev_df.copy()
                            if "리포트PDF" in html_df.columns:
                                html_df["리포트PDF"] = html_df["리포트PDF"].map(_a)
                            if "상세페이지" in html_df.columns:
                                html_df["상세페이지"] = html_df["상세페이지"].map(_a)
                            st.markdown(html_df.to_html(escape=False, index=False), unsafe_allow_html=True)
        else:
            # 폴백 표기
            st.dataframe(_show_df.drop(columns=["_row"], errors="ignore") if "_row" in _show_df else _show_df,
                         use_container_width=True, height=GRID_H)
            st.info("행 선택은 st-aggrid 설치 시 활성화됩니다.")

# ===============================
# 탭2: 종목별 검색
# ===============================
# with tab_stock:
#     st.subheader("🔎 종목별 리포트 조회")
#     col1, col2 = st.columns([2,1])
#     with col1:
#         stock_q = st.text_input("종목명(부분일치 허용)", value="", placeholder="예: 삼성전자")
#     with col2:
#         ticker_q = st.text_input("티커(정확히 6자리)", value="", placeholder="예: 005930")

#     # ✅ [추가] 기본값이 삼성전자일 때 자동으로 필터 적용되도록 보정
#     # if not stock_q.strip():
#     #     stock_q = "삼성전자"

#     stock_docs = [d for d in docs if match_stock(d, stock_q or None, ticker_q or None)]
#     st.caption(f"검색결과: {len(stock_docs)}건 (기간·증권사 필터 적용 후 종목 필터)")
with tab_stock:
    st.subheader("🔎 종목별 리포트 조회")
    col1, col2 = st.columns([2,1])
    with col1:
        stock_q = st.text_input("종목명(부분일치 허용)", value="", placeholder="예: 삼성전자")
    with col2:
        ticker_q = st.text_input("티커(정확히 6자리)", value="", placeholder="예: 005930")

    # 1) 입력된 종목명으로 후보(종목명,티커) 목록 만들기
    #    - docs에서 부분일치로 후보 수집
    candidates = []
    if stock_q.strip():
        seen = set()
        for d in docs:
            s = (d.get("stock") or "").strip()
            t = str(d.get("ticker") or "").strip()
            if s and stock_q.lower() in s.lower():
                key = (s, t)
                if key not in seen:
                    seen.add(key)
                    label = f"{s} ({t})" if t else s
                    candidates.append({"label": label, "stock": s, "ticker": t})
        # 정렬(가독성)
        candidates.sort(key=lambda x: (x["stock"], x["ticker"]))

    # 2) 후보가 여러 개면 선택 박스 제공(사용 편의 ↑)
    selected_stock = None
    selected_ticker = None
    if candidates:
      if len(candidates) == 1:
          selected_stock = candidates[0]["stock"]
          selected_ticker = candidates[0]["ticker"]
          st.session_state["last_stock_pick"] = candidates[0]["label"]
      else:
          opt_labels = [c["label"] for c in candidates]

          # ✅ 기존 선택값이 세션에 있으면 복원
          default_index = 0
          if "last_stock_pick" in st.session_state and st.session_state["last_stock_pick"] in opt_labels:
              default_index = opt_labels.index(st.session_state["last_stock_pick"])

          # ✅ 사용자가 바꾼 선택값을 세션에 저장
          pick = st.selectbox(
              "후보 종목 선택",
              options=opt_labels,
              index=default_index,
              key="cand_pick"
          )
          st.session_state["last_stock_pick"] = pick  # 최신 선택 유지

          _picked = next((c for c in candidates if c["label"] == pick), None)
          if _picked:
              selected_stock = _picked["stock"]
              selected_ticker = _picked["ticker"]


    # 3) 필터링 로직
    #    - 후보가 선택되었으면 '정확히 그 종목/티커'로 필터
    #    - 아니면 기존 부분일치/정확 매칭 로직 사용
    if selected_stock:
        stock_docs = [
            d for d in docs
            if (d.get("stock") or "").strip() == selected_stock
            and (selected_ticker is None or str(d.get("ticker") or "").strip() == selected_ticker)
        ]
        # 상단에 검색 대상 표시
        st.markdown(f"**검색 대상:** {selected_stock} ({selected_ticker or '-'})")
    else:
        stock_docs = [d for d in docs if match_stock(d, stock_q or None, ticker_q or None)]
        # 후보 수 안내(선택 전)
        if candidates:
            st.caption(f"후보 종목 {len(candidates)}개 중에서 선택하세요.")

    st.caption(f"검색결과: {len(stock_docs)}건 (기간·증권사 필터 적용 후 종목 필터)")


    if not stock_docs:
        st.info("해당 조건의 리포트가 없습니다. 종목명/티커 또는 기간/증권사를 조정해 보세요.")
    else:
        from collections import defaultdict
        # 증권사 요약
        by_broker = defaultdict(lambda: {"reports":0, "analysts":set(), "first":None, "last":None})
        for x in stock_docs:
            b = (x.get("broker") or "").strip()
            a = (x.get("analyst_name") or x.get("analyst") or "").strip()
            d0 = (x.get("report_date") or "").strip()
            by_broker[b]["reports"] += 1
            if a: by_broker[b]["analysts"].add(a)
            if d0:
                if by_broker[b]["first"] is None or d0 < by_broker[b]["first"]:
                    by_broker[b]["first"] = d0
                if by_broker[b]["last"] is None or d0 > by_broker[b]["last"]:
                    by_broker[b]["last"] = d0
        broker_rows = [{
            "증권사": b,
            "리포트수": v["reports"],
            "애널리스트수": len(v["analysts"]),
            "최초리포트일": v["first"] or "",
            "최종리포트일": v["last"] or "",
        } for b, v in by_broker.items()]
        broker_df = pd.DataFrame(broker_rows).sort_values(["리포트수","애널리스트수"], ascending=[False, False])
        st.markdown("### 1) 증권사별 요약")
        st.dataframe(broker_df, use_container_width=True)

        # 애널리스트 요약
        st.markdown("### 2) 애널리스트별 요약")
        brokers_for_select = sorted({(x.get("broker") or "").strip() for x in stock_docs if x.get("broker")})
        pick_broker = st.selectbox("증권사 선택", ["(모두)"] + brokers_for_select, index=0, key="stock_pick_broker")
        docs2 = stock_docs if pick_broker == "(모두)" else [x for x in stock_docs if (x.get("broker") or "").strip() == pick_broker]

        by_anl = defaultdict(lambda: {"reports":0, "sum_pts":0.0, "first":None, "last":None})
        for x in docs2:
            a = (x.get("analyst_name") or x.get("analyst") or "").strip()
            d0 = (x.get("report_date") or "").strip()
            pts = float(x.get("points_total") or 0.0)
            by_anl[a]["reports"] += 1
            by_anl[a]["sum_pts"] += pts
            if d0:
                if by_anl[a]["first"] is None or d0 < by_anl[a]["first"]:
                    by_anl[a]["first"] = d0
                if by_anl[a]["last"] is None or d0 > by_anl[a]["last"]:
                    by_anl[a]["last"] = d0
        anl_rows = [{
            "애널리스트": a or "(unknown)",
            "리포트수": v["reports"],
            "평균점수": round(v["sum_pts"]/v["reports"], 4) if v["reports"] else 0.0,
            "최초리포트일": v["first"] or "",
            "최종리포트일": v["last"] or "",
        } for a, v in by_anl.items()]
        anl_df = pd.DataFrame(anl_rows).sort_values(["평균점수","리포트수"], ascending=[False, False]).reset_index(drop=True)
        st.dataframe(anl_df, use_container_width=True)

        # 상세
        st.markdown("### 3) 상세 내역")
        detail_rows = []
        for r in stock_docs:
            # ✅ 제목 보강: 비어있으면 다른 키/대체문구로 보완
            title_safe = (r.get("title")
                          or r.get("pdf_title")
                          or r.get("detail_title")
                          or "").strip()
            if not title_safe:
                title_safe = f'{(r.get("stock") or "").strip()} 리포트'

            # ✅ 마지막 종가 추출
            last_close = last_close_from_horizons(r)

            detail_rows.append({
                "리포트일": r.get("report_date",""),
                "증권사": r.get("broker",""),
                "애널리스트": r.get("analyst_name") or r.get("analyst",""),
                "종목": r.get("stock",""),          # (이전 수정에서 추가되어 있을 수 있음)
                "티커": r.get("ticker",""),         # (이전 수정에서 추가되어 있을 수 있음)
                "레이팅": r.get("rating") or r.get("rating_norm",""),
                "목표가": r.get("target_price"),
                "마지막종가": last_close,            # ✅ 새 컬럼 추가 (목표가 옆)
                "제목": title_safe,                  # ✅ 빈 제목 보정
                "리포트PDF": r.get("pdf_url","") or r.get("report_url",""),
                "상세페이지": r.get("detail_url",""),
            })

        det_df = pd.DataFrame(detail_rows).sort_values("리포트일", ascending=False).reset_index(drop=True)

        if "리포트PDF" in det_df.columns:
            det_df["리포트PDF"] = det_df["리포트PDF"].map(_norm_url)
        if "상세페이지" in det_df.columns:
            det_df["상세페이지"] = det_df["상세페이지"].map(_norm_url)
        try:
            st.dataframe(
                det_df,
                use_container_width=True,
                column_config={
                    "리포트PDF": st.column_config.LinkColumn(label="리포트PDF", display_text="열기"),
                    "상세페이지": st.column_config.LinkColumn(label="상세페이지", display_text="열기"),
                }
            )
        except Exception:
            def _a(href):
                return f'<a href="{href}" target="_blank" rel="noopener noreferrer">열기</a>' if href else ""
            html_df = det_df.copy()
            if "리포트PDF" in html_df.columns:
                html_df["리포트PDF"] = html_df["리포트PDF"].map(_a)
            if "상세페이지" in html_df.columns:
                html_df["상세페이지"] = html_df["상세페이지"].map(_a)
            st.markdown(html_df.to_html(escape=False, index=False), unsafe_allow_html=True)
