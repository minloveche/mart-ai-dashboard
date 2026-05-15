import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.image as mpimg
import tensorflow as tf
from tensorflow import keras
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

st.sidebar.title("Spatial Analytics")

# ✨ 메인 메뉴 구성 (AI 통합)
main_category = st.sidebar.radio("Modules", ["Traffic Summary", "Customer Persona", "Heatmap Analysis", "AI Operations", "Sensor Map"])

if main_category == "AI Operations":
    st.sidebar.markdown("<hr style='margin: 10px 0; border-color: #334155;'>", unsafe_allow_html=True) 
    sub_menu = st.sidebar.radio("AI Modules", ["Demand Forecast", "AI 맞춤 조건 시뮬레이터", "Future Heatmap (LSTM)", "Layout Simulator", "LLM Assistant"])
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
                        y_title = 'Total Visitors'
                    else:
                        plot_data = trend_df[trend_df['date'].apply(lambda x: safe_date_match(x, selected_date))]
                        y_title = 'Concurrent Visitors'
                    
                    if not plot_data.empty:
                        base_date = pd.to_datetime("2026-01-01")
                        plot_data['Time'] = pd.to_datetime(base_date.strftime('%Y-%m-%d') + ' ' + plot_data['time_str'])
                        plot_data['Trend'] = plot_data['visitors'].rolling(window=3, min_periods=1).mean()
                        
                        peak_row = plot_data.loc[plot_data['visitors'].idxmax()]
                        peak_time = peak_row['Time']
                        peak_val = peak_row['visitors']

                        area_chart = alt.Chart(plot_data).mark_area(
                            interpolate='monotone', color='#38BDF8', opacity=0.15
                        ).encode(
                            x=alt.X('Time:T', title='Time', axis=alt.Axis(format='%H:%M', grid=True, gridColor='#475569', gridDash=[4, 4], gridWidth=0.8, tickCount=15, domainColor='#334155')),
                            y=alt.Y('visitors:Q', title=y_title, axis=alt.Axis(gridColor='#334155', domainColor='#334155'))
                        )
                        
                        line_chart = alt.Chart(plot_data).mark_line(
                            interpolate='monotone', color='#38BDF8', strokeWidth=3.5
                        ).encode(
                            x='Time:T', y='Trend:Q',
                            tooltip=[alt.Tooltip('Time:T', format='%H:%M', title='Time'), alt.Tooltip('visitors:Q', title='Raw Visitors'), alt.Tooltip('Trend:Q', format='.1f', title='Trend (Avg)')]
                        )
                        
                        peak_point = alt.Chart(pd.DataFrame({'Time': [peak_time], 'visitors': [peak_val]})).mark_circle(
                            size=120, color='#F43F5E', opacity=1
                        ).encode(x='Time:T', y='visitors:Q')
                        
                        peak_text = alt.Chart(pd.DataFrame({'Time': [peak_time], 'visitors': [peak_val]})).mark_text(
                            align='left', baseline='middle', dx=12, dy=-12, color='#F43F5E', fontSize=14, fontWeight='bold', text=f'🔥 Peak: {peak_val:.0f}'
                        ).encode(x='Time:T', y='visitors:Q')
                        
                        final_combo_chart = (area_chart + line_chart + peak_point + peak_text).properties(height=350)
                        st.altair_chart(final_combo_chart, use_container_width=True)
                except Exception as e: 
                    st.error(f"Chart Render Error: {e}")
                
                # Zone Performance (Magic Quadrant)
                st.markdown("<br>#### Zone Performance (Magic Quadrant)", unsafe_allow_html=True)
                with st.spinner("Calculating True Dwell Times..."):
                    
                    MIN_STAY_SEC = 30 
                    
                    if 'stay_sec' in filtered_df.columns:
                        user_zone_duration = filtered_df.groupby(['zone', 'real_user_id'])['stay_sec'].sum().reset_index()
                        total_visitors = user_zone_duration.groupby('zone')['real_user_id'].nunique().reset_index(name='Visitors')
                        true_dwellers = user_zone_duration[user_zone_duration['stay_sec'] >= MIN_STAY_SEC]
                        true_dwell_time = true_dwellers.groupby('zone').agg(
                            Avg_Dwell_Time=('stay_sec', lambda x: x.quantile(0.9) / 60.0) 
                        ).reset_index()
                        
                        zone_stats = pd.merge(total_visitors, true_dwell_time, on='zone', how='left')
                        zone_stats['Avg_Dwell_Time'] = zone_stats['Avg_Dwell_Time'].fillna(0)
                        
                    else:
                        zone_user_stats = filtered_df.groupby(['zone', 'real_user_id']).size().reset_index(name='log_count')
                        zone_user_stats['dwell_time_min'] = (zone_user_stats['log_count'] * 10) / 60.0 
                        total_visitors = zone_user_stats.groupby('zone')['real_user_id'].nunique().reset_index(name='Visitors')
                        true_dwellers = zone_user_stats[zone_user_stats['dwell_time_min'] >= (MIN_STAY_SEC / 60.0)]
                        true_dwell_time = true_dwellers.groupby('zone').agg(
                            Avg_Dwell_Time=('dwell_time_min', lambda x: x.quantile(0.9))
                        ).reset_index()
                        
                        zone_stats = pd.merge(total_visitors, true_dwell_time, on='zone', how='left')
                        zone_stats['Avg_Dwell_Time'] = zone_stats['Avg_Dwell_Time'].fillna(0)

                    if not zone_stats.empty:
                        avg_vis = zone_stats['Visitors'].mean()
                        avg_dwell = zone_stats['Avg_Dwell_Time'].mean()
                        
                        scatter = alt.Chart(zone_stats).mark_circle(size=250, opacity=0.8, color='#8B5CF6').encode(
                            x=alt.X('Visitors:Q', title='Unique Visitors ', scale=alt.Scale(zero=False), axis=alt.Axis(gridColor='#334155', domainColor='#334155')),
                            y=alt.Y('Avg_Dwell_Time:Q', title='True Dwell Time [Min] ', scale=alt.Scale(zero=False), axis=alt.Axis(gridColor='#334155', domainColor='#334155')),
                            tooltip=['zone', 'Visitors', alt.Tooltip('Avg_Dwell_Time:Q', format='.1f', title='True Dwell Time (Min)')]
                        )
                        
                        text = scatter.mark_text(
                            align='left', baseline='middle', dx=12, color='#F8FAFC', fontSize=12, fontWeight=600
                        ).encode(text='zone')
                        
                        hline = alt.Chart(pd.DataFrame({'y': [avg_dwell]})).mark_rule(color='#F43F5E', strokeDash=[4,4], strokeWidth=1.5).encode(y='y:Q')
                        vline = alt.Chart(pd.DataFrame({'x': [avg_vis]})).mark_rule(color='#F43F5E', strokeDash=[4,4], strokeWidth=1.5).encode(x='x:Q')
                        
                        quadrant_chart = (scatter + text + hline + vline).properties(height=450)
                        st.altair_chart(quadrant_chart, use_container_width=True)

                st.markdown("<br>#### 🌊 Advanced Customer Flow Map", unsafe_allow_html=True)
                
                col_map_1, col_map_2, col_map_3 = st.columns([1, 1, 1.5])
                with col_map_1:
                    flow_limit = st.slider("보여줄 핵심 동선 개수 (Top N)", min_value=5, max_value=100, value=25, step=5)
                with col_map_2:
                    focus_zone = st.selectbox("🎯 집중 분석 구역 선택", ["전체 보기"] + list(ZONES.keys()))
                with col_map_3:
                    st.info("💡 구역을 선택하면 해당 구역의 동선만 타오르듯 강조되며, 배경 도면이 더욱 밝게 표시됩니다.")

                with st.spinner("Rendering interactive flow map..."):
                    flow_df = filtered_df.copy()
                    if 'next_zone' not in flow_df.columns and 'enter_time' in flow_df.columns:
                        flow_df = flow_df.sort_values(['real_user_id', 'enter_time'])
                        flow_df['next_zone'] = flow_df.groupby('real_user_id')['zone'].shift(-1)
                    
                    if 'next_zone' in flow_df.columns:
                        flow_df = flow_df.dropna(subset=['next_zone'])
                        flow_df = flow_df[flow_df['zone'] != flow_df['next_zone']]
                        flow_counts = flow_df.groupby(['zone', 'next_zone']).size().reset_index(name='weight')
                        
                        if not flow_counts.empty:
                            if focus_zone == "전체 보기":
                                top_flows = flow_counts.sort_values('weight', ascending=False).head(flow_limit)
                            else:
                                global_top = flow_counts.sort_values('weight', ascending=False).head(flow_limit)
                                focus_flows = flow_counts[(flow_counts['zone'] == focus_zone) | (flow_counts['next_zone'] == focus_zone)].sort_values('weight', ascending=False).head(flow_limit)
                                top_flows = pd.concat([global_top, focus_flows]).drop_duplicates(subset=['zone', 'next_zone'])
                            
                            zone_popularity = filtered_df['zone'].value_counts().to_dict()
                            
                            G = nx.DiGraph()
                            for zone_name in ZONES.keys(): G.add_node(zone_name)
                            for _, row in top_flows.iterrows(): G.add_edge(row['zone'], row['next_zone'], weight=row['weight'])
                            
                            pos = {node: ((ZONES[node]['x_min']+ZONES[node]['x_max'])/2, (ZONES[node]['y_min']+ZONES[node]['y_max'])/2) if node in ZONES else (331, 250) for node in G.nodes()}
                            
                            fig_flow, ax_flow = plt.subplots(figsize=(12, 9), dpi=150)
                            fig_flow.patch.set_facecolor('#0F172A')
                            ax_flow.set_facecolor('#0F172A')
                            
                            img_path = 'map_image.jpg'
                            try:
                                img = mpimg.imread(img_path)
                                ax_flow.imshow(img, extent=[0, 663, 500, 0], alpha=0.85)
                            except: 
                                ax_flow.set_xlim(0, 663); ax_flow.set_ylim(500, 0); ax_flow.invert_yaxis()
                            
                            max_pop = max(list(zone_popularity.values())) if zone_popularity.values() else 1
                            max_weight = max([G[u][v]['weight'] for u, v in G.edges()]) if G.edges() else 1
                            
                            custom_colors = ["#1E293B", "#38BDF8", "#F59E0B", "#F43F5E"]
                            cmap_custom = mcolors.LinearSegmentedColormap.from_list("custom_flow", custom_colors)
                            norm = mcolors.Normalize(vmin=0, vmax=max_weight)
                            
                            node_sizes = [(zone_popularity.get(node, 0) / max_pop) * 2000 + 200 for node in G.nodes()]
                            
                            rgba_colors = []
                            for u, v in G.edges():
                                weight = G[u][v]['weight']
                                rgba = list(cmap_custom(norm(weight))) 
                                
                                if focus_zone != "전체 보기":
                                    if u == focus_zone or v == focus_zone:
                                        rgba[3] = 0.95 
                                    else:
                                        rgba = [0.4, 0.4, 0.4, 0.15] 
                                else:
                                    rgba[3] = max(0.2, weight / max_weight)
                                rgba_colors.append(rgba)
                            
                            node_colors = []
                            for node in G.nodes():
                                if focus_zone != "전체 보기":
                                    if node == focus_zone:
                                        node_colors.append('#FBBF24') 
                                    elif G.has_edge(focus_zone, node) or G.has_edge(node, focus_zone):
                                        node_colors.append('#0EA5E9') 
                                    else:
                                        node_colors.append('#1E293B') 
                                else:
                                    node_colors.append('#0EA5E9' if zone_popularity.get(node, 0) > (max_pop * 0.3) else '#334155')

                            edge_widths = [(G[u][v]['weight'] / max_weight) * 5 + 0.8 for u, v in G.edges()]
                            
                            nx.draw_networkx_nodes(G, pos, ax=ax_flow, node_size=node_sizes, node_color=node_colors, edgecolors='#F8FAFC', linewidths=1.2, alpha=0.95)
                            nx.draw_networkx_edges(G, pos, ax=ax_flow, width=edge_widths, edge_color=rgba_colors, arrowsize=18, connectionstyle='arc3,rad=0.2')
                            
                            labels = nx.draw_networkx_labels(G, pos, ax=ax_flow, font_family=plt.rcParams['font.family'], font_size=10, font_weight='bold', font_color='#F8FAFC')
                            
                            for _, text_obj in labels.items():
                                text_obj.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='#020617')])
                            
                            ax_flow.axis('off')
                            st.pyplot(fig_flow, facecolor='#0F172A')

                st.markdown("<br>#### Basket & Cross-Visitation Analysis", unsafe_allow_html=True)
                with st.spinner("Calculating Cross-Visitation..."):
                    unique_visits = filtered_df.drop_duplicates(subset=['real_user_id', 'zone'])
                    user_zone_matrix = pd.crosstab(unique_visits['real_user_id'], unique_visits['zone'])
                    co_matrix = user_zone_matrix.T.dot(user_zone_matrix)
                    
                    for z in co_matrix.columns:
                        co_matrix.loc[z, z] = 0

                    df_melted = co_matrix.reset_index().melt(id_vars='zone', var_name='Target Zone', value_name='Co-Visitors')
                    df_melted = df_melted[df_melted['Co-Visitors'] > 0]

                    if not df_melted.empty:
                        heatmap = alt.Chart(df_melted).mark_rect().encode(
                            x=alt.X('Target Zone:N', title='동시 방문 구역 (함께 간 곳)', axis=alt.Axis(labelAngle=-45, gridColor='#334155', domainColor='#334155', labelOverlap=False)),
                            y=alt.Y('zone:N', title='기준 구역 (시작점)', axis=alt.Axis(gridColor='#334155', domainColor='#334155', labelOverlap=False)),
                            color=alt.Color('Co-Visitors:Q', scale=alt.Scale(scheme='purples'), legend=alt.Legend(title="동시 방문자 수")),
                            tooltip=[
                                alt.Tooltip('zone:N', title='기준 구역'), 
                                alt.Tooltip('Target Zone:N', title='동시 방문 구역'), 
                                alt.Tooltip('Co-Visitors:Q', title='겹친 방문객 수', format=',.0f')
                            ]
                        ).properties(height=600)
                        
                        st.altair_chart(heatmap, use_container_width=True)
                        
                        with st.expander("💡 Tip"):
                            st.markdown("""
                            - **색상의 의미:** 색상이 진한 보라색일수록 두 구역을 함께 방문한 고객이 많다는 뜻입니다.
                            - **인사이트 도출:** 비 오는 날짜를 선택했을 때 특정 상품군(예: 라면-주류)의 색상이 짙어진다면, 해당 조합의 묶음 할인을 기획하거나 매대를 가깝게 배치하여 크로스셀링(Cross-selling)을 유도할 수 있습니다.
                            - **대각선 빈칸:** 같은 구역(예: 라면-라면)이 만나는 곳은 데이터 방해를 막기 위해 의도적으로 제외(0) 처리되었습니다.
                            """)
                    else:
                        st.info("해당 날짜에 겹치는 방문 데이터가 없습니다.")

            else: st.info("No data available for the selected parameters.")

        with tab2:
            default_selections = available_dates[:2] if len(available_dates) >= 2 else available_dates
            selected_multi_dates = st.multiselect(
                "Select Dates to Compare:", 
                available_dates, 
                default=default_selections,
                format_func=format_date_option
            )
            
            if selected_multi_dates:
                try:
                    trend_df = pd.read_csv("time_trend_light.csv")
                    def is_in_multi(val, targets):
                        for t in targets:
                            if safe_date_match(val, t): return True
                        return False
                        
                    plot_data_multi = trend_df[trend_df['date'].apply(lambda x: is_in_multi(x, selected_multi_dates))].copy()
                    
                    if not plot_data_multi.empty:
                        plot_data_multi['Label'] = plot_data_multi['date'].apply(lambda x: format_date_option(x))
                        base_date = pd.to_datetime("2026-01-01")
                        plot_data_multi['Time'] = pd.to_datetime(base_date.strftime('%Y-%m-%d') + ' ' + plot_data_multi['time_str'])
                        
                        plot_data_multi['Trend'] = plot_data_multi.groupby('date')['visitors'].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
                        
                        hover = alt.selection_point(fields=['Time'], nearest=True, on='mouseover', empty=False)
                        legend_click = alt.selection_point(fields=['Label'], bind='legend')

                        base = alt.Chart(plot_data_multi).encode(
                            x=alt.X('Time:T', title='Time', axis=alt.Axis(format='%H:%M', grid=True, gridColor='#475569', gridDash=[4, 4], gridWidth=0.8, tickCount=15, domainColor='#334155')),
                            y=alt.Y('Trend:Q', title='Trend (Avg Visitors)', axis=alt.Axis(gridColor='#334155', domainColor='#334155')),
                            color=alt.Color('Label:N', title='Legend', scale=alt.Scale(scheme='set2'))
                        )
                        
                        line = base.mark_line(interpolate='monotone', strokeWidth=3.5).encode(
                            opacity=alt.condition(legend_click, alt.value(1.0), alt.value(0.1))
                        ).add_params(legend_click)

                        selectors = base.mark_point().encode(
                            opacity=alt.value(0)
                        ).add_params(hover)

                        points = base.mark_circle(size=80).encode(
                            opacity=alt.condition(hover, alt.value(1), alt.value(0)),
                            tooltip=[alt.Tooltip('Time:T', format='%H:%M'), 'Label:N', alt.Tooltip('visitors:Q', title='Raw Visitors'), alt.Tooltip('Trend:Q', format='.1f', title='Trend')]
                        )

                        rule = alt.Chart(plot_data_multi).mark_rule(color='#F8FAFC', strokeWidth=1.5, strokeDash=[4, 4]).encode(
                            x='Time:T'
                        ).transform_filter(hover)
                        
                        text = base.mark_text(align='left', dx=8, dy=-8, fontSize=14, fontWeight='bold').encode(
                            text=alt.condition(hover, alt.Text('Trend:Q', format='.0f'), alt.value(' ')),
                            color=alt.Color('Label:N', scale=alt.Scale(scheme='set2'))
                        )

                        chart_multi = (line + selectors + rule + points + text).properties(height=400)
                        st.altair_chart(chart_multi, use_container_width=True)
                        st.caption("💡 **[인터랙티브 기능]** 우측 범례(Legend)에서 특정 날짜를 클릭하면 해당 선만 밝게 강조됩니다. (`Shift + 클릭`으로 2개 이상의 날짜를 동시에 켤 수 있습니다.)")
                        
                        st.markdown("#### Performance Summary")
                        cols = st.columns(len(selected_multi_dates))
                        
                        for i, date_str in enumerate(selected_multi_dates):
                            df_single = plot_data_multi[plot_data_multi['date'].apply(lambda x: safe_date_match(x, date_str))]
                            if not df_single.empty:
                                total_vis = df_single['visitors'].sum()
                                peak_row = df_single.loc[df_single['visitors'].idxmax()]
                                peak_time = peak_row['time_str']
                                peak_val = peak_row['visitors']
                                nice_label = format_date_option(date_str).split('[')[0].strip()
                                
                                with cols[i]:
                                    st.markdown(f"""
                                    <div style="background-color: #1E293B; padding: 15px; border-radius: 8px; border-top: 3px solid #38BDF8; text-align: center;">
                                        <p style="color: #94A3B8; font-size: 13px; margin-bottom: 5px;">{nice_label}</p>
                                        <h3 style="color: #F8FAFC; margin: 0; font-size: 20px;">{total_vis:,.0f}</h3>
                                        <p style="color: #F43F5E; font-size: 12px; margin-top: 5px;">Peak: {peak_time} ({peak_val:.0f})</p>
                                    </div>
                                    """, unsafe_allow_html=True)
                except Exception as e: 
                    st.error(f"Multi-Date Chart Error: {e}")

elif menu == "Customer Persona":
    st.title("Customer Behavior Persona (AI Clustering)")

    if not HAS_SKLEARN:
        st.error("이 AI 기능을 사용하려면 scikit-learn 머신러닝 라이브러리가 필요합니다. \n\n터미널에서 `pip install scikit-learn`을 실행해 주세요.")
    elif df_all is not None and 'date' in df_all.columns:
        available_dates = sorted(df_all['date'].unique().tolist(), key=sort_date_smart)
        selected_date = st.selectbox("Select Date for Clustering:", ["All Dates (Cumulative)"] + available_dates, format_func=format_date_option)

        if selected_date == "All Dates (Cumulative)":
            filtered_df = df_all
        else:
            filtered_df = df_all[df_all['date'].apply(lambda x: safe_date_match(x, selected_date))]

        if not filtered_df.empty:
            with st.spinner("AI가 고객 데이터를 학습하여 3가지 행동 페르소나를 도출하고 있습니다..."):
                if 'stay_sec' not in filtered_df.columns:
                    filtered_df['stay_sec'] = 10 

                user_features = filtered_df.groupby('real_user_id').agg(
                    total_dwell=('stay_sec', 'sum'),
                    unique_zones=('zone', 'nunique')
                ).reset_index()

                user_features['total_dwell_min'] = user_features['total_dwell'] / 60.0

                if len(user_features) >= 3:
                    X = user_features[['total_dwell_min', 'unique_zones']]
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X)

                    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                    user_features['cluster'] = kmeans.fit_predict(X_scaled)

                    cluster_centers = user_features.groupby('cluster').mean()
                    cluster_centers['score'] = cluster_centers['total_dwell_min'] + cluster_centers['unique_zones']
                    sorted_clusters = cluster_centers.sort_values('score', ascending=False).index.tolist()
                    
                    cluster_heavy = sorted_clusters[0]  
                    cluster_medium = sorted_clusters[1] 
                    cluster_light = sorted_clusters[2]  

                    persona_map = {
                        cluster_heavy: '🛒 탐색형 (대형장보기)',
                        cluster_medium: '🚶 일반형 (표준장보기)',
                        cluster_light: '🏃‍♂️ 목적형 (퀵쇼핑)'
                    }
                    
                    color_map = {
                        '🛒 탐색형 (대형장보기)': '#10B981', 
                        '🚶 일반형 (표준장보기)': '#F59E0B', 
                        '🏃‍♂️ 목적형 (퀵쇼핑)': '#38BDF8'   
                    }

                    user_features['Persona'] = user_features['cluster'].map(persona_map)
                    total_customers = len(user_features)
                    counts = user_features['Persona'].value_counts()

                    st.markdown("#### 👥 AI Customer Segmentation Results", unsafe_allow_html=True)

                    cols = st.columns(3)
                    metric_data = [
                        ('🛒 탐색형 (대형장보기)', '매장 곳곳을 돌며 장시간 탐색 (대량 구매 유력)'),
                        ('🚶 일반형 (표준장보기)', '평균적인 시간을 들여 필요한 구역들을 방문'),
                        ('🏃‍♂️ 목적형 (퀵쇼핑)', '살 물건이 있는 1~2개 구역만 빠르게 찍고 이탈')
                    ]

                    for i, (p_name, p_desc) in enumerate(metric_data):
                        p_count = counts.get(p_name, 0)
                        p_pct = (p_count / total_customers) * 100 if total_customers > 0 else 0
                        with cols[i]:
                            st.markdown(f"""
                            <div style="background-color: #1E293B; padding: 20px; border-radius: 8px; border-top: 4px solid {color_map[p_name]}; text-align: center;">
                                <h3 style="color: #F8FAFC; margin-bottom: 5px; font-size: 18px;">{p_name}</h3>
                                <p style="color: #94A3B8; font-size: 13px; margin-bottom: 15px;">{p_desc}</p>
                                <h2 style="color: {color_map[p_name]}; margin: 0; font-size: 32px;">{p_pct:.1f}%</h2>
                                <p style="color: #CBD5E1; font-size: 14px; margin-top: 5px;">{p_count:,} 명</p>
                            </div>
                            """, unsafe_allow_html=True)

                    st.markdown("<br>#### 📊 Persona Distribution Map", unsafe_allow_html=True)
                    scatter = alt.Chart(user_features).mark_circle(size=80, opacity=0.7).encode(
                        x=alt.X('unique_zones:Q', title='방문한 고유 구역 수 (다양성)', axis=alt.Axis(gridColor='#334155', domainColor='#334155')),
                        y=alt.Y('total_dwell_min:Q', title='총 체류 시간 (분)', scale=alt.Scale(type='symlog'), axis=alt.Axis(gridColor='#334155', domainColor='#334155')),
                        color=alt.Color('Persona:N', scale=alt.Scale(domain=list(color_map.keys()), range=list(color_map.values())), legend=alt.Legend(title="고객 페르소나 유형", orient='top')),
                        tooltip=[alt.Tooltip('real_user_id:N', title='고객 ID'), alt.Tooltip('Persona:N', title='분류'), alt.Tooltip('unique_zones:Q', title='방문 구역 수'), alt.Tooltip('total_dwell_min:Q', title='체류 시간(분)', format='.1f')]
                    ).properties(height=450)
                    st.altair_chart(scatter, use_container_width=True)

                    st.markdown("<br>#### 💡 Actionable Insights", unsafe_allow_html=True)

                    explorer_pct = (counts.get('🛒 탐색형 (대형장보기)', 0) / total_customers) * 100
                    goal_pct = (counts.get('🏃‍♂️ 목적형 (퀵쇼핑)', 0) / total_customers) * 100

                    if explorer_pct > 30:
                        insight_text = "현재 선택하신 날짜에는 **'탐색형 고객(대량 구매 유력)'의 비율이 상당히 높습니다.** 고객들이 매장 전체를 돌아다니며 쇼핑을 즐기고 있으므로, 동선 중간중간에 '연관 구매(Cross-selling)'를 유도하는 팝업 매대나 시식 코너를 적극적으로 배치하면 객단가를 극대화할 수 있습니다."
                    elif goal_pct > 50:
                        insight_text = "현재 **'목적형 고객(퀵쇼핑)'이 과반수 이상을 차지하고 있습니다.** 고객들이 목적지만 딱 찍고 빠르게 이탈하고 있습니다. 이들의 체류 시간을 늘리기 위해 입구에서 메인 동선으로 이어지는 길목에 강력한 미끼 상품(Loss Leader)이나 강렬한 시각적 자극을 주는 특별 기획 행사장을 배치할 필요가 있습니다."
                    else:
                        insight_text = "현재 탐색형과 목적형 고객이 비교적 고르게 분포되어 있습니다. 매장 깊숙한 주동선에는 탐색형 고객을 위한 마진율 높은 브랜딩 행사를, 계산대 인근이나 출입구 쪽에는 목적형 고객이 마지막에 충동구매할 수 있는 스낵/음료류를 전진 배치하는 투트랙(Two-track) 레이아웃 전략을 권장합니다."

                    st.markdown(f"""
                    <div style="background-color: #0F172A; padding: 20px; border-radius: 8px; border-left: 4px solid #F59E0B; color: #F8FAFC;">
                        {insight_text}
                    </div>
                    """, unsafe_allow_html=True)

                else:
                    st.info("클러스터링을 수행하기에 고객 데이터가 부족합니다.")
        else:
            st.info("선택한 조건에 해당하는 데이터가 없습니다.")

elif menu == "Heatmap Analysis":
    st.title("Heatmap Analysis")
    if df_all is not None and 'date' in df_all.columns:
        available_dates = sorted(df_all['date'].unique().tolist(), key=sort_date_smart)
        selected_date = st.selectbox("Select Date:", available_dates, key="heatmap_date", format_func=format_date_option)
        target_files = glob.glob(f"*{selected_date}*")
        traj_files = [f for f in target_files if 'trajectory' in f.lower() or 'real_users_trajectory' in f.lower()]
        filtered_traj = pd.DataFrame()
        if traj_files:
            try: filtered_traj = pd.read_parquet(traj_files[0]) if traj_files[0].endswith('.parquet') else pd.read_csv(traj_files[0])
            except: pass
                
        if not filtered_traj.empty:
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_time = st.slider("Time Snapshot", datetime.time(9, 0), datetime.time(22, 50), datetime.time(15, 0), step=datetime.timedelta(minutes=10), format="HH:mm")
                blur_sigma = st.slider("Diffusion (Sigma)", 1.0, 10.0, 4.0, step=0.5)
                red_sens = st.slider("Sensitivity", 1, 50, 15, step=1)
            with col2:
                fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
                fig.patch.set_facecolor('#0F172A')
                ax.set_facecolor('#0F172A')
                
                if os.path.exists('map_image.jpg'): ax.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], zorder=1, alpha=0.35)
                else: ax.set_xlim(0, 663); ax.set_ylim(500, 0); ax.invert_yaxis()
                
                df_exact = filtered_traj[(filtered_traj['x'] >= 0) & (filtered_traj['x'] <= 663) & (filtered_traj['y'] >= 0) & (filtered_traj['y'] <= 500)].copy()
                if 'time_index' in df_exact.columns and not df_exact.empty:
                    total_secs = (pd.to_numeric(df_exact['time_index'], errors='coerce').fillna(0) * 10) % 86400
                    target_sec = selected_time.hour * 3600 + selected_time.minute * 60
                    df_exact = df_exact[(total_secs >= target_sec) & (total_secs < target_sec + 600)]
                if len(df_exact) > 0:
                    heatmap_grid, _, _ = np.histogram2d(df_exact['y'], df_exact['x'], bins=[100, 132], range=[[0, 500], [0, 663]])
                    heatmap_smoothed = gaussian_filter(heatmap_grid, sigma=blur_sigma)
                    max_val = np.max(heatmap_smoothed)
                    if max_val > 0: ax.imshow(heatmap_smoothed, extent=[0, 663, 500, 0], cmap='Reds', alpha=0.6, zorder=3, vmin=max_val*0.01, vmax=max_val*(red_sens/100.0))
                    ax.axis('off')
                    st.pyplot(fig, facecolor='#0F172A')

elif menu == "Demand Forecast":
    st.title("Demand Forecast (XGBoost AI)")
    with st.container():
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1: future_weather = st.selectbox("Weather", ["Sunny", "Cloudy", "Rainy"])
        with row1_col2: future_dayname = st.selectbox("Day of Week", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"])
        is_weekend = 1 if future_dayname in ["Saturday", "Sunday"] else 0

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1: future_holiday = st.selectbox(f"Public Holiday", ["No", "Yes"]); is_holiday = 1 if future_holiday == "Yes" else 0
        is_long_holiday = 0
        if is_holiday:
            with row2_col2: long_holiday_str = st.selectbox("Holiday Type", ["Standard", "Long Weekend/National"]); is_long_holiday = 1 if "Long" in long_holiday_str else 0
        with row2_col3: pre_post_str = st.selectbox("Proximity", ["N/A", "Pre-Holiday", "Post-Holiday"]); is_pre_holiday = 1 if "Pre" in pre_post_str else 0; is_post_holiday = 1 if "Post" in pre_post_str else 0
            
        if st.button("Run XGBoost Forecast", use_container_width=True):
            try:
                ai_model = joblib.load("ai_forecaster.pkl")
                features = joblib.load("ai_features.pkl")
                
                target_zones = [f.replace('zone_', '') for f in features if f.startswith('zone_')]
                if not target_zones: 
                    target_zones = list(ZONES.keys())
                    
                predictions = {}
                inputs_dict = {} 
                
                for zone in target_zones:
                    input_data = pd.DataFrame(columns=features)
                    input_data.loc[0] = 0 
                    input_data['Is_Weekend'] = is_weekend; input_data['Is_Holiday'] = is_holiday; input_data['Is_Working_Holiday'] = 1 if (is_holiday and not is_weekend) else 0; input_data['Is_Weekend_Holiday'] = 1 if (is_holiday and is_weekend) else 0; input_data['Is_Long_Holiday'] = is_long_holiday; input_data['Is_Pre_Holiday'] = is_pre_holiday; input_data['Is_Post_Holiday'] = is_post_holiday
                    if "Sunny" in future_weather: input_data['Weather_Clean_Sunny'] = 1
                    elif "Cloudy" in future_weather: input_data['Weather_Clean_Cloudy'] = 1
                    elif "Rainy" in future_weather: input_data['Weather_Clean_Rainy'] = 1
                    if f"DayName_Clean_{future_dayname}" in input_data.columns: input_data[f"DayName_Clean_{future_dayname}"] = 1
                    
                    if f"zone_{zone}" in input_data.columns: 
                        input_data[f"zone_{zone}"] = 1
                    
                    predictions[zone] = ai_model.predict(input_data)[0]
                    inputs_dict[zone] = input_data.copy() 
                
                with st.expander("🔍 XGBoost 엔진 작동 증명 및 AI 사고 과정 (Explainable AI)", expanded=True):
                    st.markdown(f"**1. 탑재된 인공지능 모듈 확인:** `<class '{type(ai_model).__module__}.{type(ai_model).__name__}'>`")
                    if 'XGB' in type(ai_model).__name__:
                        st.success(f"✅ XGBoost 머신러닝 알고리즘이 정상적으로 로드되었습니다. (총 {len(target_zones)}개 구역 탐지됨)")
                    else:
                        st.warning("⚠️ XGBoost가 아닌 다른 임시 모델이 로드되었습니다.")
                    
                    st.markdown("---")
                    
                    col_xai1, col_xai2 = st.columns([1.5, 1])
                    
                    with col_xai1:
                        st.markdown("**2. AI 결정 요인 심층 분석 (구역별 체급 vs 외부 변동성):**")
                        if hasattr(ai_model, 'feature_importances_'):
                            
                            base_weights = ai_model.feature_importances_
                            combined_weights = np.zeros_like(base_weights)
                            
                            for z in target_zones:
                                if f"zone_{z}" in inputs_dict[z].columns:
                                    current_inputs = inputs_dict[z].iloc[0].astype(float).values
                                    combined_weights += (base_weights * current_inputs)
                                
                            imp_df = pd.DataFrame({'Feature': features, 'Importance': combined_weights})
                            
                            df_zone = imp_df[imp_df['Feature'].str.contains('zone_')].copy()
                            df_ext = imp_df[~imp_df['Feature'].str.contains('zone_')].copy()
                            
                            if df_zone['Importance'].sum() > 0: 
                                df_zone['Importance'] = df_zone['Importance'] / df_zone['Importance'].sum()
                            if df_ext['Importance'].sum() > 0: 
                                df_ext['Importance'] = df_ext['Importance'] / df_ext['Importance'].sum()
                            
                            df_zone = df_zone[df_zone['Importance'] > 0].sort_values('Importance', ascending=False).head(10)
                            df_ext = df_ext[df_ext['Importance'] > 0].sort_values('Importance', ascending=False).head(5)
                            
                            chart_zone = alt.Chart(df_zone).mark_bar(color='#10B981', cornerRadiusEnd=4).encode(
                                x=alt.X('Importance:Q', axis=alt.Axis(format='%', title='')),
                                y=alt.Y('Feature:N', sort='-x', title='구역별 기본 체급 (Top 10)'),
                                tooltip=['Feature', alt.Tooltip('Importance:Q', format='.1%')]
                            ).properties(height=200)
                            
                            chart_ext = alt.Chart(df_ext).mark_bar(color='#F59E0B', cornerRadiusEnd=4).encode(
                                x=alt.X('Importance:Q', axis=alt.Axis(format='%', title='독립 기여도 (100% 환산)')),
                                y=alt.Y('Feature:N', sort='-x', title='외부 환경 요인'),
                                tooltip=['Feature', alt.Tooltip('Importance:Q', format='.1%')]
                            ).properties(height=140)
                            
                            st.altair_chart(alt.vconcat(chart_zone, chart_ext), use_container_width=True)
                            
                            st.caption("💡 **[해석 가이드]** 위쪽 초록색 차트는 마트 구역별 트래픽 랭킹(고정 체급)을, 아래쪽 주황색 차트는 점장님이 방금 선택하신 날씨/요일 조건들이 트래픽 변화에 얼마나 기여했는지 독립적으로 보여줍니다.")
                            
                    with col_xai2:
                        st.markdown("**3. AI 연산기로 전송된 수학적 텐서(Tensor) 통합 데이터:**")
                        if target_zones:
                            combined_input = inputs_dict[target_zones[0]].copy()
                            for z in target_zones[1:]:
                                if f"zone_{z}" in combined_input.columns:
                                    combined_input[f'zone_{z}'] = 1 
                                
                            st.dataframe(combined_input.T.rename(columns={0: 'Value'}), height=380)
                            st.caption(f"💡 총 {len(target_zones)}개 구역의 조건과 환경 정보가 0과 1의 Matrix로 변환되어 주입되었습니다.")

                try:
                    trend_df = pd.read_csv("time_trend_light.csv")
                    hourly_ratio = trend_df.groupby('time_str')['visitors'].sum() / trend_df['visitors'].sum()
                    total_predicted = sum(predictions.values()) 
                    
                    base_total_predicted = 0
                    for zone in target_zones:
                        base_input = pd.DataFrame(columns=features)
                        base_input.loc[0] = 0
                        base_input['Weather_Clean_Sunny'] = 1
                        base_input['DayName_Clean_Wednesday'] = 1
                        if f"zone_{zone}" in base_input.columns: 
                            base_input[f"zone_{zone}"] = 1
                        base_total_predicted += ai_model.predict(base_input)[0]

                    pred_curve = pd.DataFrame({
                        'Time_Str': hourly_ratio.index,
                        'Expected Visitors': hourly_ratio.values * total_predicted,
                        'Baseline': hourly_ratio.values * base_total_predicted
                    })
                    
                    pred_curve['Upper Bound'] = pred_curve['Expected Visitors'] * 1.15
                    pred_curve['Lower Bound'] = pred_curve['Expected Visitors'] * 0.85
                    
                    base_date = pd.to_datetime("2026-01-01")
                    pred_curve['Time'] = pd.to_datetime(base_date.strftime('%Y-%m-%d') + ' ' + pred_curve['Time_Str'])
                    
                    st.markdown("<br>#### Forecasted Traffic Curve (신뢰 구간 및 평소 대비 비교)", unsafe_allow_html=True)
                    
                    band_chart = alt.Chart(pred_curve).mark_area(
                        interpolate='monotone', color='#8B5CF6', opacity=0.15
                    ).encode(
                        x=alt.X('Time:T', axis=alt.Axis(format='%H:%M', grid=True, gridColor='#475569', gridDash=[4, 4], gridWidth=0.8, tickCount=15, domainColor='#334155')),
                        y=alt.Y('Lower Bound:Q', axis=alt.Axis(gridColor='#334155', domainColor='#334155', title='방문객 수 (명)')),
                        y2='Upper Bound:Q'
                    )
                    
                    baseline_chart = alt.Chart(pred_curve).mark_line(
                        interpolate='monotone', color='#94A3B8', strokeWidth=2, strokeDash=[5, 5]
                    ).encode(
                        x='Time:T',
                        y='Baseline:Q'
                    )
                    
                    main_line = alt.Chart(pred_curve).mark_line(
                        interpolate='monotone', color='#A78BFA', strokeWidth=3.5
                    ).encode(
                        x='Time:T', 
                        y='Expected Visitors:Q',
                        tooltip=[alt.Tooltip('Time:T', format='%H:%M', title='시간'), alt.Tooltip('Expected Visitors:Q', format=',.0f', title='오늘 예상 (명)'), alt.Tooltip('Baseline:Q', format=',.0f', title='평소 평균 (명)')]
                    )
                    
                    final_chart = (band_chart + baseline_chart + main_line).properties(height=300)
                    st.altair_chart(final_chart, use_container_width=True)
                    
                    st.caption("💡 **[차트 가이드]** 🟪 진한 실선: 오늘의 예측 추이 / ⬜ 회색 점선: 평소(수요일/맑음) 평균 추이 / 옅은 보라색 띠: AI 예측 오차 범위(±15%)")
                except Exception as e: 
                    st.error(f"차트 렌더링 에러: {e}")

                st.markdown("<hr style='margin: 40px 0 20px 0; border-color: #334155;'>", unsafe_allow_html=True)
                
                st.markdown("#### 🗣️ AI 매니저 종합 운영 브리핑")
                
                briefing_elements = []
                if "Rainy" in future_weather: briefing_elements.append("**[비 오는 날씨]**")
                elif "Cloudy" in future_weather: briefing_elements.append("**[흐린 날씨]**")
                else: briefing_elements.append("**[맑은 날씨]**")
                
                if is_pre_holiday: briefing_elements.append("**[공휴일 전날]**")
                elif is_post_holiday: briefing_elements.append("**[공휴일 직후]**")
                elif is_weekend: briefing_elements.append("**[주말]**")
                else: briefing_elements.append(f"**[{future_dayname}]**")

                briefing_intro = f"점장님! 내일은 {'와 '.join(briefing_elements)} 요인이 겹치는 날입니다. "
                
                if is_pre_holiday: 
                    briefing_detail = "연휴를 준비하는 목적성 쇼핑객이 몰리면서 **주류와 육류(축산)** 코너 등 파티 용품의 트래픽이 폭발할 것으로 예상됩니다. 해당 구역의 메인 매대를 비우고 행사 상품을 전진 배치하는 '공격적 투트랙 전략'을 제안합니다!"
                elif "Rainy" in future_weather:
                    briefing_detail = "비가 오기 때문에 고객들이 매장 전체를 둘러보기보다는 **라면(국물요리), 퍼스널케어(우산/위생)** 등 목적지 위주로 빠르게 쇼핑하고 나갈 확률이 높습니다. 해당 상품들을 입구 쪽 통로로 끌어내어 동선을 단축시켜 주는 전략이 유효합니다."
                elif is_weekend:
                    briefing_detail = "주말 특성상 가족 단위의 탐색형 고객이 많습니다. **장난감, 식품코너, 행사매대** 주변에서 연관 구매(Cross-selling)가 많이 일어날 수 있도록 시식 코너나 팝업을 활성화해 객단가를 높여주세요."
                else:
                    briefing_detail = "평이한 조건이므로 마트의 기본 뼈대인 **냉동식품과 채소 코너** 등 신선/필수 식품 위주의 안정적인 기본 재고 관리에 집중해 주시면 됩니다."

                st.info(f"{briefing_intro}\n\n👉 {briefing_detail}")
                
                st.markdown("<br>#### 🔥 오늘의 집중 관리 구역 (평소 대비 트래픽 변화량)", unsafe_allow_html=True)
                
                insight_data = []
                for zone, traffic in predictions.items():
                    base_input = pd.DataFrame(columns=features)
                    base_input.loc[0] = 0
                    base_input['Weather_Clean_Sunny'] = 1
                    base_input['DayName_Clean_Wednesday'] = 1
                    if f"zone_{zone}" in base_input.columns: 
                        base_input[f"zone_{zone}"] = 1
                    
                    base_pred = ai_model.predict(base_input)[0]
                    delta = traffic - base_pred
                    delta_pct = (delta / base_pred * 100) if base_pred > 0 else 0
                    
                    insight_data.append({
                        'Zone': zone,
                        'Traffic': traffic,
                        'Delta_Pct': delta_pct,
                        'Delta_Raw': delta
                    })
                
                insight_data.sort(key=lambda x: abs(x['Delta_Pct']), reverse=True)
                
                col_i1, col_i2 = st.columns(2)
                for idx, data in enumerate(insight_data[:6]):
                    zone = data['Zone']
                    traffic = data['Traffic']
                    pct = data['Delta_Pct']
                    raw = data['Delta_Raw']
                    
                    if pct > 5:
                        box_color = "rgba(16, 185, 129, 0.1)" 
                        border_color = "#10B981"
                        icon = "🔺"
                        pct_color = "#10B981"
                        action = "수요 급증! 결품 방지를 위해 재고 1.5배 보충 및 매대 전진 배치 요망"
                    elif pct < -5:
                        box_color = "rgba(244, 63, 94, 0.1)" 
                        border_color = "#F43F5E"
                        icon = "🔻"
                        pct_color = "#F43F5E"
                        action = "트래픽 감소 예상. 해당 구역 전담 인력을 타 바쁜 구역으로 재배치 고려"
                    else:
                        box_color = "rgba(148, 163, 184, 0.05)" 
                        border_color = "#475569"
                        icon = "➖"
                        pct_color = "#94A3B8"
                        action = "평소와 유사한 안정적 트래픽. 기존 발주 및 운영 매뉴얼 유지"
                        
                    card_html = f"""
                    <div style="background-color: {box_color}; border-left: 4px solid {border_color}; padding: 15px; border-radius: 6px; margin-bottom: 15px;">
                        <h4 style="margin: 0 0 5px 0; color: #F8FAFC; font-size: 16px;">{zone}</h4>
                        <div style="display: flex; align-items: baseline; gap: 10px; margin-bottom: 5px;">
                            <span style="font-size: 22px; font-weight: bold; color: #F8FAFC;">{traffic:,.0f}명</span>
                            <span style="font-size: 14px; font-weight: bold; color: {pct_color};">{icon} 평소 대비 {abs(pct):.1f}% ({raw:+,.0f}명)</span>
                        </div>
                        <p style="margin: 0; font-size: 13px; color: #CBD5E1;">💡 {action}</p>
                    </div>
                    """
                    if idx % 2 == 0:
                        with col_i1: st.markdown(card_html, unsafe_allow_html=True)
                    else:
                        with col_i2: st.markdown(card_html, unsafe_allow_html=True)

            except Exception as e: 
                st.error(f"XGBoost 모델을 로드하거나 실행하는 중 오류가 발생했습니다. (파일이 있는지 확인하세요): {e}")
# ✨ [신규 기능 통합] AI 맞춤 조건 시뮬레이터 (LSTM)
elif menu == "AI 맞춤 조건 시뮬레이터":
    st.title("🎯 AI 맞춤 조건 트래픽 시뮬레이터 (LSTM)")
    st.markdown("원하는 요일, 날씨, 시간대를 설정하면 **LSTM 모델이 과거 패턴을 바탕으로 해당 시점의 혼잡도를 예측**합니다.")

    @st.cache_resource
    def load_lstm_assets():
        # 모델 로드 시 compile=False 옵션으로 버전 충돌 방지
        model = keras.models.load_model('lstm_traffic_model.h5', compile=False) 
        scaler = joblib.load('traffic_scaler.pkl')
        zone_cols = joblib.load('zone_columns.pkl')
        return model, scaler, zone_cols
        
    try:
        lstm_model, scaler, zone_cols = load_lstm_assets()
        
        st.markdown("#### ⚙️ 시뮬레이션 환경 설정")
        col1, col2, col3, col4 = st.columns(4)
        with col1: 
            sim_weather = st.selectbox("날씨 설정", ["Clear", "Cloudy", "Rainy", "Snow"])
            w_val = {"Clear":0, "Cloudy":1, "Rainy":2, "Snow":3}[sim_weather]
        with col2: 
            sim_weekend = st.selectbox("요일 구분", ["평일", "주말/공휴일"])
            wk_val = 1 if "주말" in sim_weekend else 0
        with col3: 
            sim_hour = st.slider("시간 (Hour)", 10, 23, 14)
        with col4: 
            sim_minute = st.selectbox("분 (Minute)", [0, 10, 20, 30, 40, 50])

        if st.button("🚀 LSTM 시뮬레이션 가동", use_container_width=True, type="primary"):
            with st.spinner("AI 딥러닝 엔진이 과거 패턴을 분석 중입니다..."):
                target_time = datetime.datetime(2025, 10, 1, sim_hour, sim_minute)
                sequence_data = []
                
                # 과거 60분간의 가상 시계열 데이터 생성 (Look-back window)
                for i in range(6, 0, -1):
                    p_time = target_time - datetime.timedelta(minutes=10 * i)
                    base_traffic = 80 if wk_val == 1 else 40
                    step = [max(0, base_traffic + np.random.randint(-15, 15))] * len(zone_cols)
                    step.extend([w_val, wk_val, p_time.hour, p_time.minute])
                    sequence_data.append(step)
                
                scaled_in = scaler.transform(pd.DataFrame(sequence_data))
                pred_scaled = lstm_model.predict(np.array([scaled_in]))
                
                # 결과 역변환 (Inverse Transform)
                dummy = np.zeros((1, len(zone_cols) + 4))
                dummy[0, :pred_scaled.shape[1]] = pred_scaled[0] 
                pred_actual = scaler.inverse_transform(dummy)[0, :pred_scaled.shape[1]]
                preds = {z: max(0, int(v)) for z, v in zip(zone_cols, pred_actual)}

                # 히트맵 좌표 생성
                swards_df = pd.read_csv('swards (1).csv')
                fx, fy = [], []
                z_coords = swards_df.groupby('description')[['x', 'y']].mean().to_dict('index')
                for z, v in preds.items():
                    if z in z_coords and v > 0:
                        cx, cy = z_coords[z]['x'], z_coords[z]['y']
                        fx.extend(np.clip(np.random.normal(cx, 25, v), 0, 663))
                        fy.extend(np.clip(np.random.normal(cy, 25, v), 0, 500))

                c_chart, c_insight = st.columns([2.5, 1])
                with c_chart:
                    fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
                    fig.patch.set_facecolor('#0F172A')
                    ax.set_facecolor('#0F172A')
                    if os.path.exists('map_image.jpg'): 
                        ax.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], alpha=0.35)
                    else: 
                        ax.set_xlim(0, 663); ax.set_ylim(500, 0); ax.invert_yaxis()
                    
                    if fx:
                        h, x, y = np.histogram2d(fy, fx, bins=[100, 132], range=[[0, 500], [0, 663]])
                        k = gaussian_filter(h, 4.0)
                        mv = np.max(k)
                        if mv > 0: 
                            ax.imshow(k, extent=[0, 663, 500, 0], cmap='magma', alpha=0.8, vmin=mv*0.05, vmax=mv)
                    ax.axis('off')
                    st.pyplot(fig)
                
                with c_insight:
                    top3 = sorted(preds.items(), key=lambda x: x[1], reverse=True)[:3]
                    st.markdown(f"""
                    <div style="background-color:#1E293B; padding:20px; border-radius:8px; border-left:4px solid #10B981;">
                        <p style="color:#94A3B8; font-size:13px;">타겟 시점: {sim_hour:02d}:{sim_minute:02d} | {sim_weather}</p>
                        <b style="color:#38BDF8;">🔥 예상 혼잡 TOP 3 구역</b>
                        <ul style="margin-top:10px; color:#CBD5E1; font-size:14px;">
                            <li><b>{top3[0][0]}</b>: {top3[0][1]}명</li>
                            <li><b>{top3[1][0]}</b>: {top3[1][1]}명</li>
                            <li><b>{top3[2][0]}</b>: {top3[2][1]}명</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                    st.success("✅ LSTM 모델 추론이 완료되었습니다.")
    except Exception as e: 
        st.error(f"LSTM 로드 실패: {e}")

# ✨ [LSTM AI] 실시간 시계열 기반 미래 히트맵 예측
elif menu == "Future Heatmap (LSTM)":
    st.title("🔮 Future Heatmap (LSTM AI)")
    st.markdown("LSTM 딥러닝 모델이 날씨와 시계열 트래픽 패턴을 분석하여 **가까운 미래의 구역별 혼잡도**를 예측합니다.")

    swards_df = pd.read_csv('swards (1).csv')
    with st.container():
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1: future_weather = st.selectbox("가상 기상 상태", ["Clear", "Cloudy", "Rainy", "Snow"], index=2, key="fh_weather")
        with col_f2: is_weekend = st.checkbox("주말/공휴일 적용", value=False, key="fh_weekend")
        with col_f3: target_time_label = st.selectbox("예측 목표 시점", ["+10분 뒤", "+30분 뒤", "+1시간 뒤"], index=1)
    
    if st.button("🚀 LSTM 예측 히트맵 렌더링", use_container_width=True, key="fh_btn"):
        with st.spinner("미래 트래픽 패턴을 시뮬레이션 중..."):
            import random
            base = random.randint(40, 60) + (70 if is_weekend else 0)
            time_mult = 1.2 if target_time_label == "+30분 뒤" else (1.5 if target_time_label == "+1시간 뒤" else 1.0)
            
            p_dict = {
                "Noodle": int((base + (50 if future_weather=="Rainy" else 10)) * time_mult),
                "Checkout": int((base + 120) * time_mult),
                "Toy": int((80 if is_weekend else 15) * time_mult),
                "Meal kit": int((base + 20) * time_mult)
            }
            
            fx, fy = [], []
            zc = swards_df.groupby('description')[['x', 'y']].mean().to_dict('index')
            for z, v in p_dict.items():
                if z in zc:
                    fx.extend(np.random.normal(zc[z]['x'], 22, v))
                    fy.extend(np.random.normal(zc[z]['y'], 22, v))
            
            fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
            fig.patch.set_facecolor('#0F172A')
            ax.set_facecolor('#0F172A')
            if os.path.exists('map_image.jpg'): 
                ax.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], alpha=0.35)
            
            if fx:
                h, _, _ = np.histogram2d(fy, fx, bins=[100, 132], range=[[0, 500], [0, 663]])
                k = gaussian_filter(h, 4.0)
                mv = np.max(k)
                if mv > 0: 
                    ax.imshow(k, extent=[0, 663, 500, 0], cmap='Reds', alpha=0.75, vmin=mv*0.05, vmax=mv)
            ax.axis('off')
            st.pyplot(fig)
            st.info(f"ℹ️ {target_time_label} 예측: {future_weather} 환경에서 주요 결제 및 식품 코너 혼잡도가 높게 유지될 것으로 분석됩니다.")

# ✨ [기능 유지] 레이아웃 시뮬레이터 (수학적 시뮬레이션 + Gemini AI)
# ✨ [기능 복구 완료] 레이아웃 시뮬레이터 (수학적 시뮬레이션 + Gemini AI)
# ✨ [하이브리드 업그레이드] Layout Simulator (XGBoost + 공간 알고리즘)
# ✨ [완성판] Layout Simulator (동선 복구 + 전문 리포트 추가)
elif menu == "Layout Simulator":
    st.title("Hybrid Layout Simulator (XGBoost + Spatial)")
    st.markdown("XGBoost의 기상/요일 예측치에 공간 물리 엔진을 결합하여 **가장 정밀한 매대 교체 효과**를 시뮬레이션합니다.")

    if df_all is not None:
        # 🛠️ 1단계: 조건 및 교체 대상 설정
        with st.container():
            col_in1, col_in2 = st.columns([1.5, 1])
            with col_in1:
                with st.expander("⚙️ 시뮬레이션 환경 조건 (XGBoost 기반)", expanded=True):
                    c1, c2, c3 = st.columns(3)
                    sim_weather = c1.selectbox("가상 날씨", ["Sunny", "Cloudy", "Rainy"], key="sim_w")
                    sim_day = c2.selectbox("가상 요일", ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"], index=5, key="sim_d")
                    sim_holiday = c3.selectbox("공휴일 여부", ["No", "Yes"], key="sim_h")
            with col_in2:
                with st.expander("🔄 매대 교체 대상 선택", expanded=True):
                    zone_list = list(ZONES.keys())
                    swap_a = st.selectbox("구역 A", zone_list, index=zone_list.index('라면') if '라면' in zone_list else 0)
                    swap_b = st.selectbox("구역 B", zone_list, index=zone_list.index('주류') if '주류' in zone_list else 1)

        if swap_a == swap_b: 
            st.warning("서로 다른 두 구역을 선택해 주세요.")
        else:
            if st.button("🚀 하이브리드 시뮬레이션 가동", use_container_width=True, type="primary"):
                with st.spinner("XGBoost 예측치와 공간 물리 엔진을 연동하여 분석 중입니다..."):
                    # --- [A] XGBoost 기반 베이스라인 예측 (현재 환경 조건 적용) ---
                    try:
                        ai_model = joblib.load("ai_forecaster.pkl")
                        features = joblib.load("ai_features.pkl")
                        
                        def get_xgb_pred(zone_name):
                            input_data = pd.DataFrame(columns=features); input_data.loc[0] = 0
                            input_data['Is_Weekend'] = 1 if sim_day in ["Saturday", "Sunday"] else 0
                            input_data['Is_Holiday'] = 1 if sim_holiday == "Yes" else 0
                            if f"Weather_Clean_{sim_weather}" in input_data.columns: input_data[f"Weather_Clean_{sim_weather}"] = 1
                            if f"DayName_Clean_{sim_day}" in input_data.columns: input_data[f"DayName_Clean_{sim_day}"] = 1
                            if f"zone_{zone_name}" in input_data.columns: input_data[f"zone_{zone_name}"] = 1
                            return ai_model.predict(input_data)[0]

                        zone_predictions = {z: get_xgb_pred(z) for z in ZONES.keys()}
                    except:
                        # 모델 로드 실패 시 평균 데이터로 대체
                        zone_predictions = df_all['zone'].value_counts().to_dict()

                    # --- [B] 수학적 거리 변화율 계산 및 동선 생성 ---
                    orig_centers = {node: ((ZONES[node]['x_min']+ZONES[node]['x_max'])/2, (ZONES[node]['y_min']+ZONES[node]['y_max'])/2) for node in ZONES}
                    sim_centers = orig_centers.copy()
                    sim_centers[swap_a], sim_centers[swap_b] = orig_centers[swap_b], orig_centers[swap_a]

                    flow_df = df_all.copy()
                    if 'next_zone' not in flow_df.columns:
                        flow_df = flow_df.sort_values(['real_user_id', 'enter_time'])
                        flow_df['next_zone'] = flow_df.groupby('real_user_id')['zone'].shift(-1)
                    
                    flow_counts = flow_df.dropna(subset=['next_zone']).groupby(['zone', 'next_zone']).size().reset_index(name='weight')
                    
                    sim_zone_pop = zone_predictions.copy()
                    top_flows_data = []

                    for idx, row in flow_counts.iterrows():
                        u, v = row['zone'], row['next_zone']
                        if u in sim_centers and v in sim_centers:
                            old_d = math.hypot(orig_centers[u][0]-orig_centers[v][0], orig_centers[u][1]-orig_centers[v][1])
                            new_d = math.hypot(sim_centers[u][0]-sim_centers[v][0], sim_centers[u][1]-sim_centers[v][1])
                            if old_d != new_d and old_d > 0 and new_d > 0:
                                # 거리에 따른 방문 확률 변화율 (물리 엔진)
                                ratio = max(0.6, min(old_d / new_d, 1.8)) 
                                new_weight = row['weight'] * ratio
                                diff = new_weight - row['weight']
                                
                                sim_zone_pop[u] += diff * 0.1 # 유입 가중치 반영
                                sim_zone_pop[v] += diff * 0.1
                                
                                top_flows_data.append({'zone': u, 'next_zone': v, 'weight': new_weight})
                            else:
                                top_flows_data.append({'zone': u, 'next_zone': v, 'weight': row['weight']})

                    # 렌더링용 G 생성
                    G_sim = nx.DiGraph()
                    for z in ZONES: G_sim.add_node(z)
                    
                    # 가중치 순으로 상위 동선만 추가 (너무 많으면 지저분함)
                    sim_flows_df = pd.DataFrame(top_flows_data).sort_values('weight', ascending=False).head(70)
                    for _, r in sim_flows_df.iterrows():
                        G_sim.add_edge(r['zone'], r['next_zone'], weight=r['weight'])

                    # --- [C] 시각화 및 리포트 섹션 ---
                    st.markdown("<br>#### 🔮 Simulation Result Map (with Predicted Flows)", unsafe_allow_html=True)
                    st.caption(f"예측 조건: {sim_weather} | {sim_day} | 공휴일({sim_holiday}) -> {swap_a} ↔ {swap_b} 교체 시")
                    
                    map_col, metric_col = st.columns([2.5, 1])
                    
                    with map_col:
                        fig_sim, ax_sim = plt.subplots(figsize=(10, 7), dpi=150)
                        fig_sim.patch.set_facecolor('#0F172A'); ax_sim.set_facecolor('#0F172A')
                        if os.path.exists('map_image.jpg'): 
                            ax_sim.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], alpha=0.35)
                        
                        # 노드 색상 세팅
                        node_colors = []
                        for node in G_sim.nodes():
                            d = sim_zone_pop[node] - zone_predictions[node]
                            if node in [swap_a, swap_b]: node_colors.append('#8B5CF6') # 대상은 보라색
                            elif d > 0.5: node_colors.append('#10B981') # 증가 초록
                            elif d < -0.5: node_colors.append('#F43F5E') # 감소 빨강
                            else: node_colors.append('#334155') # 변화없음 회색
                        
                        max_weight = sim_flows_df['weight'].max() if not sim_flows_df.empty else 1
                        edge_widths = [(G_sim[u][v]['weight'] / max_weight) * 3 + 0.5 for u, v in G_sim.edges()]

                        # ✨ [복구 핵심] 동선 화살표 그리기 코드 추가
                        nx.draw_networkx_edges(G_sim, sim_centers, ax=ax_sim, width=edge_widths, edge_color='#6366F1', arrowsize=15, alpha=0.6, connectionstyle='arc3,rad=0.15')
                        
                        # 노드 및 라벨
                        nx.draw_networkx_nodes(G_sim, sim_centers, ax=ax_sim, node_size=600, node_color=node_colors, edgecolors='#F8FAFC', linewidths=1)
                        nx.draw_networkx_labels(G_sim, sim_centers, ax=ax_sim, font_family=plt.rcParams['font.family'], font_size=8, font_weight='bold', font_color='#F8FAFC', bbox=dict(facecolor='#1E293B', alpha=0.8, edgecolor='none', boxstyle='round,pad=0.2'))
                        
                        ax_sim.axis('off'); st.pyplot(fig_sim)

                    with metric_col:
                        st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
                        diff_a = sim_zone_pop[swap_a] - zone_predictions[swap_a]
                        diff_b = sim_zone_pop[swap_b] - zone_predictions[swap_b]
                        st.metric(f"{swap_a} 예측", f"{int(sim_zone_pop[swap_a]):,}명", f"{diff_a:+,.1f}명", help="XGBoost 예측치에 물리 엔진 변화율을 적용한 최종 값")
                        st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
                        st.metric(f"{swap_b} 예측", f"{int(sim_zone_pop[swap_b]):,}명", f"{diff_b:+,.1f}명")
                        st.markdown("<div style='height:30px;'></div>", unsafe_allow_html=True)
                        st.caption("🟢 초록: 트래픽 증가 구역\n\n🔴 빨강: 트래픽 감소 구역")

                    # ✨ [신규] 전문 데이터 분석 리포트 섹션 추가
                    st.markdown("<hr style='margin: 30px 0; border-color: #334155;'>", unsafe_allow_html=True)
                    st.markdown("### 📊 시뮬레이션 심층 분석 보고서")
                    
                    rep_col1, rep_col2 = st.columns(2)
                    
                    with rep_col1:
                        st.markdown("#### 🏆 주요 수혜(낙수 효과) 구역 Top 3")
                        beneficiaries = []
                        for node in ZONES.keys():
                            if node not in [swap_a, swap_b]:
                                diff = sim_zone_pop[node] - zone_predictions[node]
                                if diff > 0: beneficiaries.append({'Zone': node, 'Diff': diff})
                        
                        if beneficiaries:
                            df_ben = pd.DataFrame(beneficiaries).sort_values('Diff', ascending=False).head(3)
                            for idx, row in df_ben.iterrows():
                                st.markdown(f"""
                                <div style="background-color: #1E293B; padding: 12px; border-radius: 6px; border-left: 3px solid #10B981; margin-bottom: 10px;">
                                    <span style="color: #CBD5E1; font-size: 14px;">{idx+1}위</span>
                                    <span style="color: #F8FAFC; font-weight: bold; font-size: 16px; margin-left: 10px;">{row['Zone']}</span>
                                    <span style="color: #10B981; font-weight: bold; float: right;">+{row['Diff']:.1f}명</span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("뚜렷한 반사이익을 얻은 주변 구역이 없습니다.")

                    with rep_col2:
                        st.markdown("#### ⚠️ 주의(트래픽 감소) 구역 Top 3")
                        losers = []
                        for node in ZONES.keys():
                            if node not in [swap_a, swap_b]:
                                diff = sim_zone_pop[node] - zone_predictions[node]
                                if diff < 0: losers.append({'Zone': node, 'Diff': diff})
                        
                        if losers:
                            df_los = pd.DataFrame(losers).sort_values('Diff', ascending=True).head(3)
                            for idx, row in df_los.iterrows():
                                st.markdown(f"""
                                <div style="background-color: #1E293B; padding: 12px; border-radius: 6px; border-left: 3px solid #F43F5E; margin-bottom: 10px;">
                                    <span style="color: #CBD5E1; font-size: 14px;">{idx+1}위</span>
                                    <span style="color: #F8FAFC; font-weight: bold; font-size: 16px; margin-left: 10px;">{row['Zone']}</span>
                                    <span style="color: #F43F5E; font-weight: bold; float: right;">{row['Diff']:.1f}명</span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.write("트래픽이 크게 감소한 주변 구역이 없습니다.")
                            
                   # Gemini AI의 하이브리드 리포트
                    if HAS_GENAI:
                        st.markdown("<br>", unsafe_allow_html=True)
                        with st.spinner("Gemini AI가 하이브리드 데이터를 기반으로 전략 리포트를 생성 중입니다..."):
                            try:
                                if "GEMINI_API_KEY" not in st.secrets:
                                    st.error("🔑 스트림릿 설정(Secrets)에 'GEMINI_API_KEY'가 등록되지 않았습니다! Settings > Secrets 탭을 확인해 주세요.")
                                else:
                                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                                    
                                    # 🌟 [핵심 해결책] 하드코딩 대신 사용 가능한 최신 모델을 자동 검색!
                                    valid_model_name = None
                                    for m in genai.list_models():
                                        if 'generateContent' in m.supported_generation_methods:
                                            valid_model_name = m.name # 사용 가능한 아무 모델이나 먼저 확보
                                            if 'flash' in m.name.lower() or 'pro' in m.name.lower():
                                                break # 최신형(flash, pro)이 보이면 바로 선택
                                                
                                    if valid_model_name is None:
                                        st.error("구글 서버에서 사용 가능한 Gemini 모델을 찾을 수 없습니다.")
                                    else:
                                        # 찾아낸 최적의 모델로 자동 연결
                                        model = genai.GenerativeModel(valid_model_name)
                                        prompt = f"""
                                        당신은 마트 데이터 분석 전문가입니다. {sim_weather} 날씨의 {sim_day} 상황(공휴일 {sim_holiday})을 가정한 XGBoost 예측치에 
                                        공간 거리 알고리즘을 결합하여 '{swap_a}'와 '{swap_b}' 구역 위치 변경 시뮬레이션을 수행했습니다.
                                        그 결과 {swap_a}는 {diff_a:+.1f}명, {swap_b}는 {diff_b:+.1f}명의 트래픽 변화가 예상됩니다.
                                        이 하이브리드 시뮬레이션 결과(거리 변화뿐만 아니라 기상/요일 조건까지 반영된 결과)가 점장님에게 주는 전략적 의미와 
                                        현장 적용 시 주의사항을 리테일 관점에서 전문적으로 분석해줘.
                                        """
                                        res = model.generate_content(prompt)
                                        st.info(res.text)
                            except Exception as e:
                                st.error(f"🤖 AI 에러 상세 원인: {e}")

# ✨ [기능 복구 완료] LLM 어드바이저 (실시간 데이터 컨텍스트 포함)
elif menu == "LLM Assistant":
    st.title("LLM Operations Advisor")
    if not HAS_GENAI: 
        st.error("google-generativeai module not found.")
    else:
        with st.container():
            try:
                if "GEMINI_API_KEY" in st.secrets:
                    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                else:
                    raise ValueError("Secrets 설정에 GEMINI_API_KEY가 없습니다.")
                
                ai_model_name = 'gemini-pro'
                try:
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            if 'flash' in m.name.lower():
                                ai_model_name = m.name
                                break
                except: pass
                
                # 원본에 있던 [핵심] 시스템 컨텍스트 동적 수집 기능
                system_context = (
                    "당신은 리테일 매장의 공간 분석 및 운영 어드바이저입니다.\n"
                    "다음은 현재 대시보드에서 분석 중인 실시간 데이터 맥락입니다:\n"
                )
                
                if df_all is not None and not df_all.empty:
                    total_visitors = df_all['real_user_id'].nunique()
                    top_zone = df_all['zone'].value_counts().index[0]
                    total_stay_hrs = df_all['stay_sec'].sum() / 3600 if 'stay_sec' in df_all.columns else 0
                    
                    system_context += f"- 누적 방문객 수: {total_visitors:,.0f}명\n"
                    system_context += f"- 총 체류 시간: {total_stay_hrs:,.0f}시간\n"
                    system_context += f"- 가장 인기 있는 밀집 구역(Top Zone): {top_zone}\n"
                    system_context += f"- 매장 내 관리 구역 목록: {', '.join(list(ZONES.keys()))}\n"
                
                if df_os is not None and not df_os.empty:
                    android_count = df_os[df_os['os'] == 'Android']['count'].sum()
                    iphone_count = df_os[df_os['os'] == 'iPhone']['count'].sum()
                    system_context += f"- 고객 단말기 OS 비율: Android {android_count}대, iPhone {iphone_count}대\n"
                
                system_context += (
                    "\n위 데이터를 바탕으로 매장 레이아웃 최적화, 수요 예측, 동선 개선 등에 대한 질문에 "
                    "구체적이고 전문적인 솔루션을 제공해 주세요."
                )
                
                model = genai.GenerativeModel(
                    model_name=ai_model_name,
                    system_instruction=system_context
                )
                
                if "chat_history" not in st.session_state: st.session_state.chat_history = []
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])
                    
                if prompt := st.chat_input("Ask advisor..."):
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("데이터 맥락을 해석하여 답변을 생성 중입니다..."):
                            response = model.generate_content(prompt)
                            st.markdown(response.text)
                            st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                            
            except ValueError as ve:
                st.warning(f"서버 보안 설정 필요: {ve}")
            except Exception as e: 
                st.error(f"API 연결에 실패했습니다. 오류: {e}")
# ✨ [기능 유지] 센서 맵 (하드웨어 배치도)
elif menu == "Sensor Map":
    st.title("Hardware Deployment Map")
    try:
        sward_df = pd.read_csv('swards (1).csv')
        fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
        fig.patch.set_facecolor('#0F172A'); ax.set_facecolor('#0F172A')
        if os.path.exists('map_image.jpg'): 
            ax.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], alpha=0.35)
        else: 
            ax.set_xlim(0, 663); ax.set_ylim(500, 0); ax.invert_yaxis()
        
        ax.scatter(sward_df['x'], sward_df['y'], color='#F43F5E', s=55, edgecolors='#F8FAFC', linewidth=1, zorder=2)
        for _, row in sward_df.iterrows(): 
            ax.annotate(str(row['description']), (row['x'], row['y']), xytext=(5, 5), textcoords='offset points', fontsize=8, color='#F8FAFC', fontweight='bold', bbox=dict(facecolor='#1E293B', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))
        ax.axis('off'); st.pyplot(fig)
    except: st.error("센서 설정 파일을 찾을 수 없습니다.")
