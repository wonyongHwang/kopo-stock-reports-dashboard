
# -*- coding: utf-8 -*-
"""
종목리포트 평가 대시보드 (A안: st.data_editor 기반, 컴포넌트/iframe 제거)
- 탭1: 🏆 종목리포트 평가 랭킹보드  → 체크박스 단일 선택 + 쿼리스트링 싱크 → 하단 상세
- 탭2: 🔎 종목별 검색                 → 체크박스 단일 선택 + 쿼리스트링 싱크 → 하단 상세
(새 탭 열기 없음)
"""

# ==== 상단 공통 부트스트랩 (모든 st.* 호출보다 먼저) ====
import os
os.environ.setdefault("STREAMLIT_CACHE_DIR", "/tmp/streamlit-cache")
os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
os.environ.setdefault("STREAMLIT_SERVER_FILE_WATCHER_TYPE", "none")

import re
import datetime as dt
from typing import List, Dict, Any
import urllib.parse as _up

import pandas as pd
import pytz
import streamlit as st
from google.cloud import firestore


# 1) 가장 먼저 page_config
st.set_page_config(page_title="한국폴리텍대학 스마트금융과", layout="wide")


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

# 쿼리스트링 유틸 (탭1: pick, 탭2: pick2)
def _build_pick_token(analyst: str, broker: str) -> str:
    return _up.quote(f"{analyst}||{broker}", safe="")

def _parse_pick_token(token: str):
    try:
        if not token:
            return None
        s = _up.unquote(token)
        analyst, broker = s.split("||", 1)
        return analyst, broker
    except Exception:
        return None

def _get_qp(name: str) -> str | None:
    v = st.query_params.get(name)
    if isinstance(v, list):
        return v[0] if v else None
    return v

def _set_qp(name: str, value: str | None):
    qp = dict(st.query_params)
    if value is None:
        qp.pop(name, None)
    else:
        qp[name] = value
    st.query_params.clear()
    st.query_params.update(qp)


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

def horizon_to_rows(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Horizon 레코드를 테이블용 행 리스트로 변환.
    - target_price / price 샘플에 None 또는 문자열이 섞여 있어도 안전하게 처리.
    """
    def _to_float(x):
        if x is None:
            return None
        try:
            if isinstance(x, str):
                xs = x.strip().replace(',', '')
                if xs == '':
                    return None
                return float(xs)
            return float(x)
        except Exception:
            return None

    rows = []
    horizons = rec.get("horizons") or []
    rating_norm = rec.get("rating_norm") or rec.get("rating")
    target_price_raw = rec.get("target_price")
    target_price = _to_float(target_price_raw)

    for h in horizons:
        # 방향/수익률/정답
        desired = None
        try:
            dtg = h.get("direction_target") or {}
            desired = int(dtg.get("desired")) if dtg.get("desired") is not None else None
        except Exception:
            desired = None

        ret = h.get("return_pct", None)
        ret = _to_float(ret)
        actual_sign = (1 if ret and ret > 0 else (-1 if ret and ret < 0 else (0 if ret == 0 else None)))
        correct = None
        if desired is not None and actual_sign is not None:
            if desired == 0: correct = (actual_sign == 0)
            else:            correct = (desired == actual_sign)

        # 가격 샘플 통계
        sample = h.get("price_sample") or []
        closes = [_to_float(p.get("close")) for p in sample if p.get("close") is not None]
        closes = [c for c in closes if c is not None]
        max_c = max(closes) if closes else None
        min_c = min(closes) if closes else None
        last_c = closes[-1] if closes else None

        # 파생 지표(목표가 대비)
        target_return_pct = None
        if target_price not in (None, 0, 0.0) and last_c is not None:
            try:
                target_return_pct = round((last_c - target_price) / target_price * 100.0, 3)
            except Exception:
                target_return_pct = None

        max_reach_pct = None
        if target_price not in (None, 0, 0.0) and max_c is not None:
            try:
                max_reach_pct = round((max_c / target_price - 1.0) * 100.0, 3)
            except Exception:
                max_reach_pct = None

        rows.append({
            "보고서일": rec.get("report_date",""),
            "horizon(일)": h.get("horizon_days"),
            "예측포지션": ("상승(buy)" if desired == 1 else ("하락(sell)" if desired == -1 else "중립(hold)")),
            "레이팅": rating_norm,
            "예측목표가": target_price if target_price is not None else rec.get("target_price"),
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
# 단일선택 테이블 렌더 (체크박스 + 세션 상태 + 쿼리파라미터 싱크)
# -----------------------------
def render_single_select_table(df: pd.DataFrame, key: str, page_size: int, qp_key: str,
                               height: int = TABLE_SCROLL_HEIGHT, column_config: dict | None = None):
    """체크박스 1개만 True 되도록 강제하고, 선택이 바뀌면 쿼리파라미터(qp_key)를 업데이트 후 rerun."""
    if df is None or df.empty:
        st.info("표시할 데이터가 없습니다.")
        return None, None

    total = len(df)
    total_pages = (total + page_size - 1) // page_size
    page = st.number_input("Page", 1, max(1, total_pages), 1, key=f"{key}_page")
    start, end = (page - 1) * page_size, min(page * page_size, total)
    st.caption(f"Page {page}/{total_pages} · Rows {start+1}-{end} / {total}")

    view = df.iloc[start:end].reset_index(drop=True).copy()

    # 전역 선택 인덱스 관리
    sel_key = f"{key}__gidx"
    sel_gidx = st.session_state.get(sel_key, -1)

    # 쿼리파라미터 복원 우선
    parsed = _parse_pick_token(_get_qp(qp_key))
    if parsed:
        an, br = parsed
        # 전역 인덱스 탐색
        try:
            gidx = next(i for i, r in df.iterrows() if str(r.get("Analyst", r.get("애널리스트",""))) == an and str(r.get("Broker", r.get("증권사",""))) == br)
            sel_gidx = gidx
            st.session_state[sel_key] = sel_gidx
        except StopIteration:
            pass

    # 현재 페이지 체크 상태 구성
    pick_col = "__pick__"
    flags = []
    for i in range(len(view)):
        gidx = start + i
        flags.append(gidx == sel_gidx)
    view.insert(0, pick_col, flags)

    edited = st.data_editor(
        view,
        hide_index=True,
        use_container_width=True,
        height=height,
        disabled=False,  # 체크 가능
        column_config=column_config or {},
        key=f"{key}_editor_{page}",
    )

    # 체크된 행 판독
    true_idxs = [i for i, v in enumerate(list(edited[pick_col])) if bool(v)]

    new_sel_gidx = sel_gidx
    if len(true_idxs) == 0:
        pass  # 이전 선택 유지
    elif len(true_idxs) == 1:
        new_sel_gidx = start + true_idxs[0]
    else:
        prev_local = sel_gidx - start
        candidate = next((i for i in true_idxs if i != prev_local), true_idxs[0])
        new_sel_gidx = start + candidate

    if new_sel_gidx != sel_gidx:
        st.session_state[sel_key] = new_sel_gidx
        # 쿼리파라미터 동기화
        row = df.iloc[new_sel_gidx]
        analyst = str(row.get("Analyst", row.get("애널리스트","")))
        broker  = str(row.get("Broker",  row.get("증권사","")))
        _set_qp(qp_key, _build_pick_token(analyst, broker))
        st.rerun()

    # 반환: 현재 페이지 DF(선택 컬럼 제거), 선택한 행 dict
    edited_no_pick = edited.drop(columns=[pick_col], errors="ignore")
    picked_row = None
    if 0 <= st.session_state.get(sel_key, -1) < len(df):
        picked_row = df.iloc[st.session_state[sel_key]].to_dict()
    return edited_no_pick, picked_row


# -----------------------------
# Streamlit UI
# -----------------------------
st.title("📊 종목리포트 분석")

# --- Tabs 스타일 ---
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

    metric = st.radio("Ranking metric", ["avg", "sum"], index=0, horizontal=True, key="metric")
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
# 탭1: 랭킹보드 — 단일 선택 + 하단 상세
# ===============================
with tab_rank:
    st.subheader("🏆 Top Analysts")
    rank_df = make_rank_table(docs, metric=st.session_state.get("metric","avg"), min_reports=min_reports)

    if rank_df.empty:
        st.info("조건에 맞는 데이터가 없습니다. 기간/브로커/최소 리포트 수를 조정하세요.")
    else:
        per_page = st.selectbox("Rows per page", [10, 25, 50, 100], index=1, key="rank_rows_per_page")

        view, picked = render_single_select_table(
            rank_df.reset_index(drop=True), key="rank_table",
            page_size=per_page, qp_key="pick",
            height=TABLE_SCROLL_HEIGHT,
            column_config=None
        )

        if picked:
            picked_name = picked.get("Analyst", "Unknown Analyst")
            picked_broker = picked.get("Broker", "Unknown Broker")
            st.markdown(f"### {picked_name} — {picked_broker}")
            st.write(f"**Reports:** {picked.get('Reports', '-') } | **RankScore:** {picked.get('RankScore', '-') }")
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

                wanted = [
                    "보고서일", "종목", "티커", "레이팅", "예측포지션", "예측목표가",
                    "horizon(일)",
                    "구간최고종가", "구간최저종가", "구간마지막종가", "목표가대비최대도달%",
                    "리포트일에 매수했을경우 수익률%", "방향정답", "방향점수", "목표근접점수", "목표가HIT",
                    "리포트총점", "리포트PDF", "상세페이지"
                ]
                cols = [c for c in wanted if c in ev_df.columns] + [c for c in ev_df.columns if c not in wanted]
                ev_df = ev_df[cols]

                st.data_editor(
                    ev_df,
                    hide_index=True,
                    use_container_width=True,
                    height=TABLE_SCROLL_HEIGHT,
                    disabled=True,
                    column_config={
                        "리포트PDF": st.column_config.LinkColumn(label="리포트PDF", display_text="열기"),
                        "상세페이지": st.column_config.LinkColumn(label="상세페이지", display_text="열기"),
                    },
                    key="rank_evidence_table"
                )
        else:
            st.info("표에서 한 행을 체크하면 아래에 상세가 표시됩니다.")

# ===============================
# 탭2: 종목별 검색 — 단일 선택 + 하단 상세
# ===============================
with tab_stock:
    st.subheader("🔎 종목별 리포트 조회")
    stock_q = st.text_input("종목명(부분일치 허용)", key="stock_q", placeholder="(공란=전체)")
    ticker_q = st.text_input("티커(정확히 6자리)", key="ticker_q", placeholder="예: 005930")

    stock_docs = [d for d in docs if match_stock(d, stock_q or None, ticker_q or None)] if (stock_q or ticker_q) else docs
    st.caption(f"검색결과: {len(stock_docs)}건 (기간·증권사 필터 적용 후 종목 필터)")

    if not stock_docs:
        st.info("해당 조건의 리포트가 없습니다. 종목명/티커 또는 기간/증권사를 조정해 보세요.")
    else:
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

        page_size2 = st.selectbox("페이지 크기", options=[25, 50, 100, 200], index=1, key="detail_page_size_v2")
        det_view, det_pick = render_single_select_table(
            det_df, key="stock_detail_main_table_safe",
            page_size=page_size2, qp_key="pick2",
            height=TABLE_SCROLL_HEIGHT,
        )

        if det_pick:
            st.markdown("---")
            st.markdown("#### 📌 선택 리포트 상세")

            p_stock  = det_pick.get("종목","")
            p_broker = det_pick.get("증권사","")
            p_anl    = det_pick.get("애널리스트","")
            p_date   = det_pick.get("리포트일","")
            p_ticker = ""
            for _r in stock_docs:
                t = str(_r.get("ticker") or "").strip()
                s = (_r.get("stock") or "").strip()
                if s == p_stock and t:
                    p_ticker = t
                    break

            c1, c2, c3, c4 = st.columns([2,2,2,2])
            c1.markdown(f"**종목/티커**: {p_stock} ({p_ticker or '-'})")
            c2.markdown(f"**증권사**: {p_broker}")
            c3.markdown(f"**애널리스트**: {p_anl}")
            c4.markdown(f"**리포트일**: {p_date}")

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

                rank_score = (total_pts / reports_n) if reports_n else 0.0
                cov_first = min(dates) if dates else "-"
                cov_last  = max(dates) if dates else "-"
                st.write(f"**Reports:** {reports_n}  |  **RankScore(평균):** {round(rank_score, 4) if reports_n else 0.0}  |  **Coverage:** {cov_first} → {cov_last}")

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

                st.data_editor(
                    ev_df,
                    hide_index=True,
                    use_container_width=True,
                    height=TABLE_SCROLL_HEIGHT,
                    disabled=True,
                    column_config={
                        "리포트PDF": st.column_config.LinkColumn(label="리포트PDF", display_text="열기"),
                        "상세페이지": st.column_config.LinkColumn(label="상세페이지", display_text="열기"),
                    },
                    key="stock_detail_evidence_table_v2"
                )
            else:
                st.info("선택한 애널리스트의 상세 근거가 없습니다.")
        else:
            st.info("상세 목록에서 한 행을 체크하면, 아래에 해당 애널리스트의 상세가 표시됩니다.")
