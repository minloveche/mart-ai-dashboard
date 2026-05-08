import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
import platform
import re
import glob
import altair as alt
import datetime
import joblib
import networkx as nx
import math
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.patheffects as PathEffects

# --- [AI 및 ML 라이브러리 체크] ---
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

try:
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

# --- [1. 기본 설정 및 다크모드 폰트] ---
st.set_page_config(page_title="Retail Spatial Analytics", layout="wide")

alt.theme.enable('dark')
plt.style.use('dark_background')

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
    if os.path.exists(font_path):
        fm.fontManager.addfont(font_path)
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
    else:
        plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

custom_css = """
<style>
    .stApp { background-color: #0F172A; color: #F8FAFC; }
    [data-testid="stSidebar"] { background-color: #020617 !important; border-right: 1px solid #1E293B; }
    [data-testid="stSidebar"] * { color: #F8FAFC !important; }
    h1, h2, h3, h4 { color: #F8FAFC !important; font-weight: 700 !important; letter-spacing: -0.5px; }
    p, span, div { color: #CBD5E1; }
    [data-testid="stMetric"], [data-testid="stVerticalBlockBorderWrapper"], div[data-testid="stContainer"] { 
        background-color: #1E293B !important; padding: 20px; border-radius: 8px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.4); border: 1px solid #334155 !important; text-align: center; 
    }
    [data-testid="stMetricLabel"] { font-size: 14px; color: #94A3B8 !important; font-weight: 500; }
    [data-testid="stMetricValue"] { font-size: 32px; color: #38BDF8 !important; font-weight: 700; }
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { font-size: 16px; font-weight: 600; color: #64748B; padding: 10px 20px; }
    .stTabs [aria-selected="true"] { color: #38BDF8; border-bottom-color: #38BDF8; }
    div[data-baseweb="popover"] > div, ul[data-baseweb="menu"] { background-color: #1E293B !important; border: 1px solid #334155 !important; }
    li[role="option"] { background-color: #1E293B !important; color: #F8FAFC !important; }
    li[role="option"]:hover, li[aria-selected="true"] { background-color: #334155 !important; color: #38BDF8 !important; }
    div[data-baseweb="select"] > div { background-color: #0F172A !important; border-color: #334155 !important; color: #F8FAFC !important; }
    div[data-baseweb="select"] span { color: #F8FAFC !important; }
    [data-testid="stExpander"] { background-color: #1E293B; border: 1px solid #334155; border-radius: 8px; }
    [data-testid="stExpander"] summary p { font-weight: 600; color: #38BDF8; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

ZONES = {
    '행사(1)': {'x_min': 489, 'x_max': 528, 'y_min': 301, 'y_max': 374}, '문구(1)': {'x_min': 528, 'x_max': 587, 'y_min': 303, 'y_max': 372},
    '장난감': {'x_min': 494, 'x_max': 560, 'y_min': 398, 'y_max': 485}, '침구': {'x_min': 420, 'x_max': 494, 'y_min': 396, 'y_max': 482},
    '보수용품': {'x_min': 239, 'x_max': 421, 'y_min': 397, 'y_max': 493}, '음료': {'x_min': 183, 'x_max': 239, 'y_min': 397, 'y_max': 452},
    '주류': {'x_min': 99,  'x_max': 183, 'y_min': 389, 'y_max': 452}, '식품코너': {'x_min': 42,  'x_max': 102, 'y_min': 313, 'y_max': 406},
    '과자': {'x_min': 127, 'x_max': 186, 'y_min': 301, 'y_max': 374}, '화장품': {'x_min': 487, 'x_max': 586, 'y_min': 163, 'y_max': 267},
    '반찬/소스': {'x_min': 302, 'x_max': 372, 'y_min': 161, 'y_max': 223}, '커피/차': {'x_min': 208, 'x_max': 285, 'y_min': 266, 'y_max': 300},
    '주방용품': {'x_min': 298, 'x_max': 393, 'y_min': 300, 'y_max': 373}, '자동차용품': {'x_min': 389, 'x_max': 415, 'y_min': 298, 'y_max': 376},
    '문구(2)': {'x_min': 420, 'x_max': 468, 'y_min': 298, 'y_max': 374}, '냉동식품': {'x_min': 128, 'x_max': 189, 'y_min': 163, 'y_max': 299},
    '퍼스널케어': {'x_min': 371, 'x_max': 469, 'y_min': 161, 'y_max': 230}, '축산': {'x_min': 59,  'x_max': 105, 'y_min': 169, 'y_max': 297},
    '수산': {'x_min': 61,  'x_max': 159, 'y_min': 73,  'y_max': 138}, '속옷': {'x_min': 463, 'x_max': 536, 'y_min': 56,  'y_max': 135},
    '스포츠': {'x_min': 603, 'x_max': 633, 'y_min': 57,  'y_max': 137}, '스포츠(2)': {'x_min': 537, 'x_max': 602, 'y_min': 57,  'y_max': 137},
    '제임스딘': {'x_min': 429, 'x_max': 451, 'y_min': 73,  'y_max': 137}, '곡물/건조식품': {'x_min': 293, 'x_max': 426, 'y_min': 71,  'y_max': 137},
    '채소/계란/과일': {'x_min': 158, 'x_max': 292, 'y_min': 81,  'y_max': 138}, '라면': {'x_min': 209, 'x_max': 305, 'y_min': 161, 'y_max': 227},
    '행사(2)': {'x_min': 207, 'x_max': 284, 'y_min': 223, 'y_max': 265}, '시리얼': {'x_min': 286, 'x_max': 307, 'y_min': 229, 'y_max': 295},
    '휴지': {'x_min': 207, 'x_max': 294, 'y_min': 302, 'y_max': 375}, '홈데코': {'x_min': 236, 'x_max': 322, 'y_min': 399, 'y_max': 493}
}

@st.cache_data
def load_all_sessions():
    files = glob.glob("Zone_Visit_Sessions*.*") + glob.glob("sessions_compressed.*")
    files = [f for f in files if f.endswith('.parquet') or f.endswith('.csv')]
    if not files: return None
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f)
            match = re.search(r'(\d{4})[-_](\d{1,2})[-_](\d{1,2})', f)
            if match:
                y, m, d = match.groups()
                df['date'] = f"{y}-{int(m):02d}-{int(d):02d}"
            dfs.append(df)
        except: pass
    return pd.concat(dfs, ignore_index=True) if dfs else None

@st.cache_data
def load_weather():
    weather_dict = {}
    if os.path.exists("Day_Weather_Enhanced.csv"):
        try:
            df_w = pd.read_csv("Day_Weather_Enhanced.csv")
            for index, row in df_w.iterrows():
                date_str = str(row['Date']).strip()
                try: day_num = int(date_str.split('-')[-1])
                except: day_num = index + 1 
                weather = str(row['Weather']).strip()
                weather_dict[day_num] = f"{date_str} [{weather}]"
        except: pass
    return weather_dict

@st.cache_data
def load_os_summary():
    if os.path.exists("os_summary.csv"):
        try:
            return pd.read_csv("os_summary.csv")
        except:
            return None
    return None

df_all = load_all_sessions()
weather_info = load_weather()
df_os = load_os_summary()

def safe_date_match(val, target):
    if '-' in str(val) and '-' in str(target): return str(val).strip() == str(target).strip()
    def get_day_num(x):
        nums = re.findall(r'\d+', str(x).split('.')[0])
        return int(nums[-1]) if nums else None
    v1 = get_day_num(val)
    v2 = get_day_num(target)
    if v1 is not None and v2 is not None: return v1 == v2
    return str(val).strip() == str(target).strip()

def sort_date_smart(d):
    nums = re.findall(r'\d+', str(d))
    if not nums: return 99999999
    if len(nums) >= 3: return int(f"{nums[0]}{int(nums[1]):02d}{int(nums[2]):02d}")
    return int(nums[-1])

def format_date_option(d):
    if d == "All Dates (Cumulative)": return d
    try:
        day_num = int(str(d).split('-')[-1])
        return weather_info.get(day_num, str(d))
    except: return str(d)

# --- [사이드바 메뉴 설정] ---
st.sidebar.title("Spatial Analytics")

main_category = st.sidebar.radio("Modules", ["Traffic Summary", "Heatmap Analysis", "AI Operations", "Sensor Map"])

if main_category == "AI Operations":
    st.sidebar.markdown("<hr style='margin: 10px 0; border-color: #334155;'>", unsafe_allow_html=True) 
    # ✨ Customer Persona를 AI Operations 하위 메뉴로 이동
    sub_menu = st.sidebar.radio("AI Modules", ["Customer Persona", "Demand Forecast", "Layout Simulator", "LLM Assistant"])
    menu = sub_menu 
else:
    menu = main_category

if menu == "Traffic Summary":
    st.title("Traffic Summary")
    
    if df_all is not None and 'date' in df_all.columns:
        available_dates = sorted(df_all['date'].unique().tolist(), key=sort_date_smart)
        tab1, tab2 = st.tabs(["Single Date Analysis", "Multi-Date Comparison"])
        
        with tab1:
            selected_date = st.selectbox("Select Date:", ["All Dates (Cumulative)"] + available_dates, format_func=format_date_option)
            if selected_date == "All Dates (Cumulative)":
                filtered_df = df_all
            else:
                filtered_df = df_all[df_all['date'].apply(lambda x: safe_date_match(x, selected_date))]
                
            if not filtered_df.empty:
                total_users = df_all.groupby('date')['real_user_id'].nunique().sum() if selected_date == "All Dates (Cumulative)" else filtered_df['real_user_id'].nunique()
                col1, col2, col3 = st.columns(3)
                total_stays = filtered_df['stay_sec'].sum() / 3600
                top_zone = filtered_df['zone'].value_counts().index[0]
                col1.metric("Total Visitors", f"{total_users:,.0f}")
                col2.metric("Total Dwell Time (Hrs)", f"{total_stays:,.0f}")
                col3.metric("Top Zone", top_zone)
                
                if df_os is not None:
                    if selected_date == "All Dates (Cumulative)":
                        android_count = df_os[df_os['os'] == 'Android']['count'].sum()
                        iphone_count = df_os[df_os['os'] == 'iPhone']['count'].sum()
                    else:
                        os_filtered = df_os[df_os['date'].apply(lambda x: safe_date_match(x, selected_date))]
                        android_count = os_filtered[os_filtered['os'] == 'Android']['count'].sum()
                        iphone_count = os_filtered[os_filtered['os'] == 'iPhone']['count'].sum()
                        
                    total_os = android_count + iphone_count
                    if total_os > 0:
                        android_pct = (android_count / total_os) * 100
                        iphone_pct = (iphone_count / total_os) * 100
                        
                        st.markdown("<br>#### Device OS Distribution", unsafe_allow_html=True)
                        st.markdown(f"""
                        <div style="background-color: #1E293B; padding: 15px 25px; border-radius: 8px; border: 1px solid #334155; margin-bottom: 10px;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                                <span style="color: #38BDF8; font-weight: 700; font-size: 15px;">🤖 Android {android_pct:.1f}% <span style="color:#94A3B8; font-weight:500;">({int(android_count):,})</span></span>
                                <span style="color: #F8FAFC; font-weight: 700; font-size: 15px;">🍏 iPhone {iphone_pct:.1f}% <span style="color:#94A3B8; font-weight:500;">({int(iphone_count):,})</span></span>
                            </div>
                            <div style="width: 100%; background-color: #334155; border-radius: 10px; height: 14px; display: flex; overflow: hidden; box-shadow: inset 0 1px 3px rgba(0,0,0,0.2);">
                                <div style="width: {android_pct}%; background-color: #38BDF8; height: 100%; transition: width 0.5s ease-in-out;"></div>
                                <div style="width: {iphone_pct}%; background-color: #F8FAFC; height: 100%; transition: width 0.5s ease-in-out;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                st.markdown("<br>#### Time-Series Traffic (Advanced Trend Analysis)", unsafe_allow_html=True)
                try:
                    trend_df = pd.read_csv("time_trend_light.csv")
                    if selected_date == "All Dates (Cumulative)":
                        plot_data = trend_df.groupby('time_str')['visitors'].sum().reset_index()
                    else:
                        plot_data = trend_df[trend_df['date'].apply(lambda x: safe_date_match(x, selected_date))]
                    
                    if not plot_data.empty:
                        base_date = pd.to_datetime("2026-01-01")
                        plot_data['Time'] = pd.to_datetime(base_date.strftime('%Y-%m-%d') + ' ' + plot_data['time_str'])
                        plot_data['Trend'] = plot_data['visitors'].rolling(window=3, min_periods=1).mean()
                        
                        area_chart = alt.Chart(plot_data).mark_area(interpolate='monotone', color='#38BDF8', opacity=0.15).encode(
                            x=alt.X('Time:T', axis=alt.Axis(format='%H:%M', gridColor='#475569', gridDash=[4, 4])),
                            y=alt.Y('visitors:Q', axis=alt.Axis(gridColor='#334155'))
                        )
                        line_chart = alt.Chart(plot_data).mark_line(interpolate='monotone', color='#38BDF8', strokeWidth=3.5).encode(x='Time:T', y='Trend:Q')
                        st.altair_chart((area_chart + line_chart).properties(height=350), use_container_width=True)
                except: pass
                
                # --- [고객 동선 맵 (Advanced Ver)] ---
                st.markdown("<br>#### 🌊 Advanced Customer Flow Map", unsafe_allow_html=True)
                
                col_map_1, col_map_2, col_map_3 = st.columns([1, 1, 1.5])
                with col_map_1: flow_limit = st.slider("보여줄 핵심 동선 개수 (Top N)", 5, 100, 25, 5)
                with col_map_2: focus_zone = st.selectbox("🎯 집중 분석 구역 선택", ["전체 보기"] + list(ZONES.keys()))
                with col_map_3: st.info("💡 구역을 선택하면 해당 구역의 동선만 강조되며, 배경 도면이 밝게 표시됩니다.")

                with st.spinner("Rendering flow map..."):
                    flow_df = filtered_df.copy()
                    if 'next_zone' not in flow_df.columns:
                        flow_df = flow_df.sort_values(['real_user_id', 'enter_time'])
                        flow_df['next_zone'] = flow_df.groupby('real_user_id')['zone'].shift(-1)
                    
                    flow_df = flow_df.dropna(subset=['next_zone'])
                    flow_df = flow_df[flow_df['zone'] != flow_df['next_zone']]
                    flow_counts = flow_df.groupby(['zone', 'next_zone']).size().reset_index(name='weight')
                    
                    if not flow_counts.empty:
                        top_flows = flow_counts.sort_values('weight', ascending=False).head(flow_limit)
                        if focus_zone != "전체 보기":
                            focus_flows = flow_counts[(flow_counts['zone'] == focus_zone) | (flow_counts['next_zone'] == focus_zone)].sort_values('weight', ascending=False).head(flow_limit)
                            top_flows = pd.concat([top_flows, focus_flows]).drop_duplicates(subset=['zone', 'next_zone'])
                        
                        zone_popularity = filtered_df['zone'].value_counts().to_dict()
                        G = nx.DiGraph()
                        for zone_name in ZONES.keys(): G.add_node(zone_name)
                        for _, row in top_flows.iterrows(): G.add_edge(row['zone'], row['next_zone'], weight=row['weight'])
                        pos = {node: ((ZONES[node]['x_min']+ZONES[node]['x_max'])/2, (ZONES[node]['y_min']+ZONES[node]['y_max'])/2) if node in ZONES else (331, 250) for node in G.nodes()}
                        
                        fig_flow, ax_flow = plt.subplots(figsize=(12, 9), dpi=150)
                        fig_flow.patch.set_facecolor('#0F172A')
                        ax_flow.set_facecolor('#0F172A')
                        
                        try:
                            img = mpimg.imread('map_image.jpg')
                            ax_flow.imshow(img, extent=[0, 663, 500, 0], alpha=0.85) # 배경 밝게
                        except: ax_flow.set_xlim(0, 663); ax_flow.set_ylim(500, 0); ax_flow.invert_yaxis()
                        
                        max_pop = max(list(zone_popularity.values())) if zone_popularity.values() else 1
                        max_weight = max([G[u][v]['weight'] for u, v in G.edges()]) if G.edges() else 1
                        cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_flow", ["#1E293B", "#38BDF8", "#F59E0B", "#F43F5E"])
                        norm = mcolors.Normalize(vmin=0, vmax=max_weight)
                        
                        node_sizes = [(zone_popularity.get(node, 0) / max_pop) * 2000 + 200 for node in G.nodes()]
                        node_colors, rgba_colors = [], []
                        
                        for node in G.nodes():
                            if focus_zone != "전체 보기":
                                if node == focus_zone: node_colors.append('#FBBF24')
                                elif G.has_edge(focus_zone, node) or G.has_edge(node, focus_zone): node_colors.append('#0EA5E9')
                                else: node_colors.append('#1E293B')
                            else: node_colors.append('#0EA5E9' if zone_popularity.get(node, 0) > (max_pop * 0.3) else '#334155')

                        for u, v in G.edges():
                            weight = G[u][v]['weight']
                            rgba = list(cmap_custom(norm(weight)))
                            if focus_zone != "전체 보기":
                                rgba[3] = 0.95 if (u == focus_zone or v == focus_zone) else 0.15
                                if not (u == focus_zone or v == focus_zone): rgba[:3] = [0.4, 0.4, 0.4]
                            else: rgba[3] = max(0.2, weight / max_weight)
                            rgba_colors.append(rgba)

                        nx.draw_networkx_nodes(G, pos, ax=ax_flow, node_size=node_sizes, node_color=node_colors, edgecolors='#F8FAFC', linewidths=1.2, alpha=0.95)
                        nx.draw_networkx_edges(G, pos, ax=ax_flow, width=[(G[u][v]['weight']/max_weight)*5+0.8 for u,v in G.edges()], edge_color=rgba_colors, arrowsize=18, connectionstyle='arc3,rad=0.2')
                        labels = nx.draw_networkx_labels(G, pos, ax=ax_flow, font_family=plt.rcParams['font.family'], font_size=10, font_weight='bold', font_color='#F8FAFC')
                        for _, t in labels.items(): t.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='#020617')])
                        ax_flow.axis('off')
                        st.pyplot(fig_flow, facecolor='#0F172A')

elif menu == "Customer Persona":
    st.title("Customer Behavior Persona (AI Clustering)")
    if not HAS_SKLEARN: st.error("pip install scikit-learn 필요")
    elif df_all is not None:
        available_dates = sorted(df_all['date'].unique().tolist(), key=sort_date_smart)
        selected_date = st.selectbox("Clustering Date:", ["All Dates (Cumulative)"] + available_dates, format_func=format_date_option)
        filtered_df = df_all if selected_date == "All Dates (Cumulative)" else df_all[df_all['date'].apply(lambda x: safe_date_match(x, selected_date))]
        
        if not filtered_df.empty:
            with st.spinner("AI 분석 중..."):
                user_features = filtered_df.groupby('real_user_id').agg(total_dwell=('stay_sec', 'sum'), unique_zones=('zone', 'nunique')).reset_index()
                user_features['total_dwell_min'] = user_features['total_dwell'] / 60.0
                if len(user_features) >= 3:
                    X = StandardScaler().fit_transform(user_features[['total_dwell_min', 'unique_zones']])
                    user_features['cluster'] = KMeans(n_clusters=3, random_state=42, n_init=10).fit_predict(X)
                    
                    # ✨ 개선된 라벨링 로직
                    centers = user_features.groupby('cluster').mean()
                    centers['score'] = centers['total_dwell_min'] + centers['unique_zones']
                    sorted_clusters = centers.sort_values('score', ascending=False).index.tolist()
                    
                    persona_map = {sorted_clusters[0]: '🛒 탐색형 (대형장보기)', sorted_clusters[1]: '🚶 일반형 (표준장보기)', sorted_clusters[2]: '🏃‍♂️ 목적형 (퀵쇼핑)'}
                    color_map = {'🛒 탐색형 (대형장보기)': '#10B981', '🚶 일반형 (표준장보기)': '#F59E0B', '🏃‍♂️ 목적형 (퀵쇼핑)': '#38BDF8'}
                    user_features['Persona'] = user_features['cluster'].map(persona_map)
                    
                    st.markdown("#### 👥 AI Customer Segmentation Results")
                    counts = user_features['Persona'].value_counts()
                    cols = st.columns(3)
                    persona_info = [('🛒 탐색형 (대형장보기)', '매장 전체 탐색 (큰손)'), ('🚶 일반형 (표준장보기)', '평균적인 쇼핑 (든든한 허리)'), ('🏃‍♂️ 목적형 (퀵쇼핑)', '필요한 것만 빠르게 (스나이퍼)')]
                    for i, (p_name, p_desc) in enumerate(persona_info):
                        p_count = counts.get(p_name, 0)
                        p_pct = (p_count / len(user_features)) * 100
                        with cols[i]: st.markdown(f'<div style="background-color:#1E293B; padding:20px; border-radius:8px; border-top:4px solid {color_map[p_name]}; text-align:center;"><h3 style="font-size:18px;">{p_name}</h3><p style="color:#94A3B8; font-size:13px;">{p_desc}</p><h2 style="color:{color_map[p_name]}; font-size:32px;">{p_pct:.1f}%</h2><p style="font-size:14px;">{p_count:,}명</p></div>', unsafe_allow_html=True)
                    
                    scatter = alt.Chart(user_features).mark_circle(size=80, opacity=0.7).encode(
                        x=alt.X('unique_zones:Q', title='방문 구역 수'), y=alt.Y('total_dwell_min:Q', title='체류 시간(분)', scale=alt.Scale(type='symlog')),
                        color=alt.Color('Persona:N', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=alt.Legend(orient='top'))
                    ).properties(height=450)
                    st.altair_chart(scatter, use_container_width=True)
                else: st.info("데이터 부족")

elif menu == "Heatmap Analysis":
    st.title("Heatmap Analysis")
    if df_all is not None:
        selected_date = st.selectbox("Date:", sorted(df_all['date'].unique().tolist(), key=sort_date_smart), format_func=format_date_option)
        traj_files = [f for f in glob.glob(f"*{selected_date}*") if 'trajectory' in f.lower()]
        if traj_files:
            filtered_traj = pd.read_parquet(traj_files[0]) if traj_files[0].endswith('.parquet') else pd.read_csv(traj_files[0])
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_time = st.slider("Time Snapshot", datetime.time(9, 0), datetime.time(22, 50), datetime.time(15, 0), step=datetime.timedelta(minutes=10))
                blur_sigma = st.slider("Diffusion", 1.0, 10.0, 4.0)
            with col2:
                fig, ax = plt.subplots(figsize=(10, 7))
                try: ax.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], alpha=0.35)
                except: ax.set_xlim(0, 663); ax.set_ylim(500, 0); ax.invert_yaxis()
                
                df_exact = filtered_traj[(filtered_traj['x']>=0)&(filtered_traj['x']<=663)&(filtered_traj['y']>=0)&(filtered_traj['y']<=500)].copy()
                target_sec = selected_time.hour*3600 + selected_time.minute*60
                df_exact = df_exact[((df_exact['time_index']*10)%86400 >= target_sec) & ((df_exact['time_index']*10)%86400 < target_sec+600)]
                if not df_exact.empty:
                    h, _, _ = np.histogram2d(df_exact['y'], df_exact['x'], bins=[100, 132], range=[[0, 500], [0, 663]])
                    ax.imshow(gaussian_filter(h, sigma=blur_sigma), extent=[0, 663, 500, 0], cmap='Reds', alpha=0.6, vmin=np.max(h)*0.01)
                ax.axis('off')
                st.pyplot(fig, facecolor='#0F172A')

elif menu == "Demand Forecast":
    st.title("Demand Forecast")
    try:
        model = joblib.load("ai_forecaster.pkl"); features = joblib.load("ai_features.pkl")
        col1, col2 = st.columns(2)
        weather = col1.selectbox("Weather", ["Sunny", "Cloudy", "Rainy"])
        dayname = col2.selectbox("Day", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        if st.button("Forecast"):
            st.info("예측 결과: 모델에 입력된 특성에 기반한 트래픽 예측 곡선이 출력됩니다.")
    except: st.error("Model Not Found")

elif menu == "Layout Simulator":
    st.title("Layout Simulator")
    zone_list = list(ZONES.keys())
    col1, col2 = st.columns(2)
    swap_a = col1.selectbox("Zone A", zone_list, index=0)
    swap_b = col2.selectbox("Zone B", zone_list, index=1)
    if st.button("Simulate"):
        st.success(f"{swap_a}와 {swap_b}의 위치를 변경했을 때의 예상 트래픽 변화를 분석합니다.")

elif menu == "LLM Assistant":
    st.title("LLM Operations Advisor")
    if HAS_GENAI:
        if prompt := st.chat_input("마트 운영에 대해 질문하세요"):
            with st.chat_message("user"): st.write(prompt)
            with st.chat_message("assistant"): st.write("Gemini AI가 데이터를 분석하여 답변을 생성합니다.")

elif menu == "Sensor Map":
    st.title("Sensor Deployment Map")
    try:
        sward_df = pd.read_csv('swards (1).csv')
        fig, ax = plt.subplots(figsize=(10, 7))
        try: ax.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], alpha=0.35)
        except: ax.set_xlim(0, 663); ax.set_ylim(500, 0); ax.invert_yaxis()
        ax.scatter(sward_df['x'], sward_df['y'], color='#F43F5E', s=50, edgecolors='white')
        ax.axis('off')
        st.pyplot(fig, facecolor='#0F172A')
    except: st.error("Sensor File Error")
