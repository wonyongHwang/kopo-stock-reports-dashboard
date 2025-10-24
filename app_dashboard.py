# -*- coding: utf-8 -*-
"""
종목리포트 평가 대시보드 (A안: st.data_editor 기반, 컴포넌트/iframe 제거)
- 탭1: 🏆 종목리포트 평가 랭킹보드
- 탭2: 🔎 종목별 검색
"""

# ==== 상단 공통 부트스트랩 (모든 st.* 호출보다 먼저) ====
import os
os.environ.setdefault("STREAMLIT_CACHE_DIR", "/tmp/streamlit-cache")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import re
import math
import time
import datetime as dt
from typing import List, Dict, Any

import pandas as pd
import pytz
import streamlit as st
from google.cloud import firestore


# 1) 가장 먼저 page_config
st.set_page_config(page_title="한국폴리텍대학 스마트금융과", layout="wide")


# 2) 안전 테이블 렌더 헬퍼 (AgGrid 대체)
def render_safe_table(
    df: pd.DataFrame,
    key: str,
    page_size: int = 25,
    selectable: bool = True,
    auto_pick_first: bool = True,
    height: int | None = None,
    column_config: dict | None = None,
):
    """
    - data_editor 기반 페이지네이션 + (옵션) 단일행 선택(체크박스)
    - 반환: (현재 페이지 DF, 선택행(dict)|None)
    """
    if df is None or df.empty:
        st.info("표시할 데이터가 없습니다.")
        return None, None

    total = len(df)
    total_pages = (total + page_size - 1) // page_size
    page = st.number_input("Page", 1, max(1, total_pages), 1, key=f"{key}_page")
    start = (page - 1) * page_size
    end = min(start + page_size, total)
    st.caption(f"Page {page}/{total_pages} · Rows {start+1}-{end} / {total}")

    view = df.iloc[start:end].copy().reset_index(drop=True)

    picked_row = None
    if selectable:
        pick_col = "__pick__"
        if pick_col not in view.columns:
            view.insert(0, pick_col, False)
        if auto_pick_first and len(view) > 0:
            view.at[0, pick_col] = True

        edited = st.data_editor(
            view,
            key=f"{key}_editor_{page}",
            hide_index=True,
            use_container_width=True,
            height=height,
            column_config=column_config or {},
        )
        sel_df = edited[edited[pick_col] == True].drop(columns=[pick_col], errors="ignore")
        if not sel_df.empty:
            picked_row = sel_df.iloc[0].to_dict()
        return edited.drop(columns=[pick_col], errors="ignore"), picked_row

    else:
        st.data_editor(
            view,
            key=f"{key}_editor_{page}",
            hide_index=True,
            use_container_width=True,
            height=height,
            disabled=True,
            column_config=column_config or {},
        )
        return view, None


# ====== 유틸 ======
TABLE_SCROLL_HEIGHT = 520
KST = pytz.timezone("Asia/Seoul")

def naver_item_url(ticker: str | None) -> str:
    t = (ticker or "").strip()
    if not t or len(t) != 6:
        return "https://finance.naver.com"
    return f"https://finance.naver.com/item/main.naver?code={t}"

def last_close_from_horizons(rec: dict) -> float | None:
    horizons = rec.get("horizons") or []
    if not horizons:
        return None
    try:
        horizons = sorted(horizons, key=lambda h: h.get("end_date",""))
    except Exception:
        pass
    last_close = None
    for h in horizons:
        ps = h.get("price_sample") or []
        closes = [p.get("close") for p in ps if p.get("close") is not None]
        if closes:
            last_close = float(closes[-1])
    return last_close

def normalize_broker_name(name: str) -> str:
    if not name:
        return ""
    name = re.sub(r"\s+", "", name)
    name = re.sub(r"[&/\.#\[\]]", "_", name)
    name = re.sub(r"[\(\)\[\]]", "", name)
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
    # 사용 중단(호환 위해 남김): 이제 render_safe_table이 선택 처리
    return True

def _aggrid_pick_first(sel):
    return None


# -----------------------------
# Firestore Client
# -----------------------------
@st.cache_resource(show_spinner=False)
def get_db():
    return firestore.Client()

@st.cache_data(show_spinner=False, ttl=300)
def fetch_titles_for_records(records: list[dict]) -> dict[str, str]:
    db = get_db()
    ids = []
    for r in records:
        rid = str(r.get("report_id") or "").strip()
        if rid:
            ids.append(rid)
    ids = sorted(set(ids))
    if not ids:
        return {}

    refs = [db.collection("analyst_reports").document(rid) for rid in ids]
    try:
        snaps = db.get_all(refs)
    except Exception:
        return {}

    out = {}
    for s in snaps:
        if not s.exists:
            continue
        d = s.to_dict() or {}
        t = (d.get("title") or d.get("report_title") or d.get("subject") or "").strip()
        if t:
            out[s.id] = t
    return out

def _pick_title(rec: dict) -> str:
    for k in ["title", "report_title", "subject", "report_subject", "headline", "name"]:
        t = (rec.get(k) or "").strip()
        if t:
            return t
    return ""

def _title_from_map_or_fields(rec: dict, title_map: dict[str, str] | None = None) -> str:
    rid = str(rec.get("report_id") or "").strip()
    if title_map and rid in title_map:
        return title_map[rid]
    t = _pick_title(rec)
    if t:
        return t
    stock = (rec.get("stock") or "").strip()
    rdate = (rec.get("report_date") or "").strip()
    return f"{stock or '리포트'} ({rdate or '-'})"


# -----------------------------
# 데이터 로드
# -----------------------------
@st.cache_data(show_spinner=False, ttl=60)
def load_analyst_docs(date_from: dt.date, date_to: dt.date,
                      brokers: List[str] | None, limit: int = 3000) -> List[Dict[str, Any]]:
    db = get_db()
    docs_raw = []
    server_filtered = False
    try:
        q = db.collection_group("by_analyst")
        want_min = date_from.isoformat() if date_from else None
        want_max_excl = (date_to + dt.timedelta(days=1)).isoformat() if date_to else None

        q = db.collection_group("by_analyst")
        if want_min:     q = q.where("report_date", ">=", want_min)
        if want_max_excl:q = q.where("report_date", "<",  want_max_excl)
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
        if broker_norm == "한국IR협의회" or "평가" in broker_norm:
            continue
        if broker_set and broker_norm not in broker_set:
            continue
        x["broker"] = broker_norm
        raw_name = (x.get("analyst_name") or x.get("analyst") or "").strip()
        x["analyst_name"] = raw_name
        x["analyst_name_norm"] = x.get("analyst_name_norm") or normalize_analyst_name(raw_name)

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

        if x.get("points_total") is None and x.get("horizons"):
            x["points_total"] = sum(float(h.get("points_total_this_horizon") or 0.0) for h in x.get("horizons") or [])
        out.append(x)
    return out

@st.cache_data(show_spinner=False, ttl=60)
def load_detail_for_analyst(analyst_name: str, broker: str,
                            date_from: dt.date | None, date_to: dt.date | None,
                            limit: int = 800) -> List[Dict[str, Any]]:
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
        try:
            pts = float(x.get("points_total") or 0.0)
        except Exception:
            pts = 0.0
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
        if v["n"] < min_reports: 
            continue
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
        ret = h.get("return_pct", None)
        actual_sign = _sign(ret)
        correct = None
        if desired is not None and actual_sign is not None:
            if desired == 0: correct = (actual_sign == 0)
            else:            correct = (desired == actual_sign)

        sample = h.get("price_sample") or []
        max_c, min_c, last_c = _price_stats(sample)

        target_return_pct = None
        if target_price and target_price not in ("", 0, 0.0) and last_c:
            try:
                target_return_pct = round((float(last_c)-float(target_price))/float(target_price)*100, 3)
            except Exception:
                target_return_pct = None

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
st.title("📊 종목리포트 분석")

# --- Tabs 스타일 (기존 유지) ---
st.markdown("""
<style>
div[data-testid="stTabs"] div[role="tablist"] {
  border-bottom: 2px solid #e9ecef; gap: 8px; padding-bottom: 2px; margin-bottom: 8px;
}
div[data-testid="stTabs"] button[role="tab"] {
  font-weight: 600; font-size: 0.95rem; color: #495057;
  background: #f8f9fa; border: 1px solid #e9ecef; border-bottom: none;
  border-top-left-radius: 8px; border-top-right-radius: 8px;
  padding: 10px 14px; box-shadow: inset 0 -3px 0 0 rgba(0,0,0,0.03);
}
div[data-testid="stTabs"] button[role="tab"]:hover { background: #f1f3f5; color: #343a40; }
div[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
  background: #ffffff; color: #212529; border-color: #dee2e6; box-shadow: inset 0 -4px 0 0 #0d6efd;
}
div[data-testid="stTabs"] > div[data-baseweb="tab-panel"] {
  border: 1px solid #dee2e6; border-top: none; border-radius: 0 10px 10px 10px;
  padding: 16px 16px 10px 16px; background: #ffffff;
}
div[data-testid="stTabs"] button[role="tab"] p { line-height: 1.1; margin: 0; }
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
# 오늘 범위 신선도 배너 + 탭2 병합 데이터 준비
# -----------------------------
today = dt.date.today()
is_today_range = (date_from and date_to and date_from <= today <= date_to)

if is_today_range:
    by_cnt = sum(1 for d in docs if str(d.get("report_date","")).startswith(today.isoformat()))
    db = get_db()
    ar_cnt = len(list(db.collection("analyst_reports")
                      .where("pred_time", ">=", today.isoformat())
                      .where("pred_time", "<",  (today + dt.timedelta(days=1)).isoformat())
                      .stream()))
    if ar_cnt > by_cnt:
        st.warning(f"오늘 데이터 동기화 진행 중: by_analyst {by_cnt}건 / analyst_reports {ar_cnt}건")

docs_for_stock = docs[:]
if is_today_range:
    db = get_db()
    iso_min = date_from.isoformat()
    iso_max_excl = (date_to + dt.timedelta(days=1)).isoformat()
    try:
        q_ar = (db.collection("analyst_reports")
                  .where("pred_time", ">=", iso_min)
                  .where("pred_time", "<",  iso_max_excl)
                  .limit(max_docs))
        rows = [s.to_dict() or {} for s in q_ar.stream()]
    except Exception:
        rows = []

    seen = set((
        str(x.get("report_id") or ""),
        normalize_broker_name(x.get("broker") or ""),
        (x.get("analyst_name") or x.get("analyst") or "").strip(),
        str(x.get("report_date") or "")
    ) for x in docs_for_stock)

    for x in rows:
        rd = (x.get("report_date") or x.get("pred_time") or "")
        x["report_date"] = str(rd)[:10] if rd else ""
        b = normalize_broker_name(x.get("broker") or "")
        if b == "한국IR협의회":
            continue
        x["broker"] = b
        a_raw = (x.get("analyst_name") or x.get("analyst") or "").strip()
        x["analyst_name"] = a_raw
        x["analyst_name_norm"] = x.get("analyst_name_norm") or normalize_analyst_name(a_raw)
        k = (str(x.get("report_id") or ""), b, a_raw, x["report_date"])
        if k in seen:
            continue
        seen.add(k)
        docs_for_stock.append(x)

# -----------------------------
# Tabs
# -----------------------------
tab_rank, tab_stock = st.tabs(["🏆 종목리포트 평가 랭킹보드", "🔎 종목별 검색"])

# ===============================
# 탭1: 랭킹보드 (페이지네이션 + 단일 선택)
# ===============================
with tab_rank:
    st.subheader("🏆 Top Analysts")
    rank_df = make_rank_table(docs, metric=metric, min_reports=min_reports)

    if rank_df.empty:
        st.info("조건에 맞는 데이터가 없습니다. 기간/브로커/최소 리포트 수를 조정하세요.")
    else:
        per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1, key="rank_rows_per_page")

        # 데이터 에디터로 교체
        _show_df = rank_df.reset_index(drop=True).copy()
        rank_view, rank_pick = render_safe_table(
            _show_df,
            key="rank_table",
            page_size=per_page,
            selectable=True,
            auto_pick_first=False,
            height=None,
        )

        if not rank_pick:
            st.info("상단 표에서 체크박스로 행을 선택하세요.")
        else:
            picked_name = rank_pick.get("Analyst", "Unknown Analyst")
            picked_broker = rank_pick.get("Broker", "Unknown Broker")
            st.markdown(f"### {picked_name} — {picked_broker}")
            st.write(f"**Reports:** {rank_pick.get('Reports', '-') } | **RankScore:** {rank_pick.get('RankScore', '-') }")
            st.write(f"**Coverage:** {rank_pick.get('FirstReport', '-') } → {rank_pick.get('LastReport', '-') }")

            with st.spinner("Loading evidence..."):
                items = load_detail_for_analyst(picked_name, picked_broker, date_from, date_to)

            if not items:
                st.info("상세 데이터가 없습니다.")
            else:
                evidence_rows: List[Dict[str, Any]] = []
                for it in items:
                    evidence_rows.extend(horizon_to_rows(it))
                ev_df = pd.DataFrame(evidence_rows)

                wanted = [
                    "보고서일", "종목", "티커", "레이팅", "예측포지션", "예측목표가",
                    "horizon(일)",
                    "구간최고종가", "구간최저종가", "구간마지막종가", "목표가대비최대도달%",
                    "리포트일에 매수했을경우 수익률%", "방향정답", "방향점수", "목표근접점수", "목표가HIT",
                    "리포트총점", "리포트PDF", "상세페이지"
                ]
                cols = [c for c in wanted if c in ev_df.columns] + [c for c in ev_df.columns if c not in wanted]
                ev_df = ev_df[cols]

                colcfg = {
                    "리포트PDF": st.column_config.LinkColumn(label="리포트PDF", display_text="열기"),
                    "상세페이지": st.column_config.LinkColumn(label="상세페이지", display_text="열기"),
                }
                # 상세표는 선택 불필요 → disabled 모드
                render_safe_table(
                    ev_df, key="rank_evidence_table",
                    page_size=50, selectable=False, height=TABLE_SCROLL_HEIGHT,
                    column_config=colcfg
                )

# ===============================
# 탭2: 종목별 검색
# ===============================
if "stock_q" not in st.session_state:
    st.session_state["stock_q"] = ""
if "ticker_q" not in st.session_state:
    st.session_state["ticker_q"] = ""

with tab_stock:
    st.subheader("🔎 종목별 리포트 조회")
    col1, col2 = st.columns([2,1])

    stock_q = st.text_input("종목명(부분일치 허용)", key="stock_q", placeholder="예: 삼성전자")
    ticker_q = st.text_input("티커(정확히 6자리)", key="ticker_q", placeholder="예: 005930")

    candidates = []
    if stock_q.strip():
        seen = set()
        for d in docs_for_stock:
            s = (d.get("stock") or "").strip()
            t = str(d.get("ticker") or "").strip()
            if s and stock_q.lower() in s.lower():
                key = (s, t)
                if key not in seen:
                    seen.add(key)
                    label = f"{s} ({t})" if t else s
                    candidates.append({"label": label, "stock": s, "ticker": t})
        candidates.sort(key=lambda x: (x["stock"], x["ticker"]))

    selected_stock = None
    selected_ticker = None
    if candidates:
        if len(candidates) == 1:
            selected_stock = candidates[0]["stock"]
            selected_ticker = (candidates[0]["ticker"] or "").strip() or None
            st.session_state["last_stock_pick"] = candidates[0]["label"]
        else:
            opt_labels = [c["label"] for c in candidates]
            default_index = 0
            if "last_stock_pick" in st.session_state and st.session_state["last_stock_pick"] in opt_labels:
                default_index = opt_labels.index(st.session_state["last_stock_pick"])
            pick = st.selectbox("후보 종목 선택", options=opt_labels, index=default_index, key="cand_pick")
            st.session_state["last_stock_pick"] = pick
            _picked = next((c for c in candidates if c["label"] == pick), None)
            if _picked:
                selected_stock = _picked["stock"]
                selected_ticker = (_picked["ticker"] or "").strip() or None

    if selected_stock:
        _sel_tk = (selected_ticker or "").strip()
        stock_docs = [
            d for d in docs_for_stock
            if (d.get("stock") or "").strip() == selected_stock
            and (not _sel_tk or str(d.get("ticker") or "").strip() == _sel_tk)
        ]
        st.markdown(f"**검색 대상:** {selected_stock} ({selected_ticker or '-'})")
    else:
        if not stock_q.strip() and not ticker_q.strip():
            stock_docs = docs_for_stock
            st.caption("전체 리포트를 검색 중입니다.")
        else:
            stock_docs = [d for d in docs_for_stock if match_stock(d, stock_q or None, ticker_q or None)]
            if candidates:
                st.caption(f"후보 종목 {len(candidates)}개 중에서 선택하세요.")

    st.caption(f"검색결과: {len(stock_docs)}건 (기간·증권사 필터 적용 후 종목 필터)")

    if not stock_docs:
        st.info("해당 조건의 리포트가 없습니다. 종목명/티커 또는 기간/증권사를 조정해 보세요.")
    else:
        from collections import defaultdict

        # 1) 증권사 요약
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
        render_safe_table(broker_df, key="stock_broker_summary", page_size=25, selectable=False, height=None)

        # 2) 애널리스트 요약
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
        render_safe_table(anl_df, key="stock_anl_summary", page_size=25, selectable=False, height=None)

        # 3) 상세 내역
        st.markdown("### 3) 상세 내역")
        detail_rows = []
        title_map_stock = fetch_titles_for_records(stock_docs)
        for r in stock_docs:
            title_safe = _title_from_map_or_fields(r, title_map_stock) or f'{(r.get("stock") or "").strip()} 리포트'
            try:
                last_close = last_close_from_horizons(r)
            except Exception:
                last_close = None
            detail_rows.append({
                "리포트일": r.get("report_date",""),
                "증권사": r.get("broker",""),
                "애널리스트": r.get("analyst_name") or r.get("analyst",""),
                "종목": r.get("stock",""),
                "레이팅": r.get("rating") or r.get("rating_norm",""),
                "목표가": r.get("target_price"),
                "마지막종가": last_close,
                "제목": title_safe,
            })
        det_df = pd.DataFrame(detail_rows).sort_values("리포트일", ascending=False).reset_index(drop=True)

        if det_df.empty:
            st.info("상단 표에서 행을 선택하세요.")
        else:
            page_size = st.selectbox("상세 내역 페이지 크기", options=[25, 50, 100, 200], index=1, key="detail_page_size")
            det_view, det_pick = render_safe_table(
                det_df,
                key="stock_detail_main_table_safe",
                page_size=page_size,
                selectable=True,
                auto_pick_first=True,
                height=TABLE_SCROLL_HEIGHT,
            )

            if det_pick:
                st.markdown("---")
                st.markdown("#### 📌 선택 리포트 상세")

                p_stock  = det_pick.get("종목","")
                p_ticker = ""
                # 티커 추정(원본 docs에서 종목명 매칭)
                for _r in stock_docs:
                    t = str(_r.get("ticker") or "").strip()
                    s = (_r.get("stock") or "").strip()
                    if s == p_stock and t:
                        p_ticker = t
                        break
                p_broker = det_pick.get("증권사","")
                p_anl    = det_pick.get("애널리스트","")
                p_date   = det_pick.get("리포트일","")

                c1, c2, c3, c4 = st.columns([2,2,2,2])
                c1.markdown(f"**종목/티커**: {p_stock} ({p_ticker or '-'})")
                c2.markdown(f"**증권사**: {p_broker}")
                c3.markdown(f"**애널리스트**: {p_anl}")
                c4.markdown(f"**리포트일**: {p_date}")

                st.markdown("#### 📄 해당 애널리스트의 근거 내역")
                with st.spinner("애널리스트 상세 로딩..."):
                    items = load_detail_for_analyst(p_anl, p_broker, date_from, date_to)
                    reports_n = len(items)
                    total_pts = 0.0
                    dates = []
                    for it in items:
                        try:
                            total_pts += float(it.get("points_total") or 0.0)
                        except Exception:
                            pass
                        rd = (it.get("report_date") or "").strip()
                        if rd: dates.append(rd)

                    rank_score = (total_pts / reports_n) if reports_n and metric == "avg" else total_pts
                    cov_first = min(dates) if dates else "-"
                    cov_last  = max(dates) if dates else "-"
                    st.write(f"**Reports:** {reports_n}  |  **RankScore:** {round(rank_score, 4) if reports_n else 0.0}  |  **Coverage:** {cov_first} → {cov_last}")

                if items:
                    rows_ev = []
                    for it in items:
                        rows_ev.extend(horizon_to_rows(it))
                    ev_df = pd.DataFrame(rows_ev)

                    wanted_cols = [
                        "보고서일", "종목", "티커", "레이팅", "예측포지션", "예측목표가",
                        "horizon(일)",
                        "구간최고종가", "구간최저종가", "구간마지막종가", "목표가대비최대도달%",
                        "리포트일에 매수했을경우 수익률%", "방향정답", "방향점수", "목표근접점수", "목표가HIT",
                        "리포트총점", "리포트PDF", "상세페이지"
                    ]
                    cols = [c for c in wanted_cols if c in ev_df.columns] + [c for c in ev_df.columns if c not in wanted_cols]
                    ev_df = ev_df[cols]

                    colcfg = {
                        "리포트PDF": st.column_config.LinkColumn(label="리포트PDF", display_text="열기"),
                        "상세페이지": st.column_config.LinkColumn(label="상세페이지", display_text="열기"),
                    }
                    render_safe_table(
                        ev_df, key="stock_detail_evidence_table",
                        page_size=50, selectable=False, height=TABLE_SCROLL_HEIGHT,
                        column_config=colcfg
                    )
                else:
                    st.info("선택한 애널리스트의 상세 근거가 없습니다.")

                st.markdown("#### 📈 종목 정보")
                st.markdown(f"[네이버 금융 상세 보기]({naver_item_url(p_ticker)})")
