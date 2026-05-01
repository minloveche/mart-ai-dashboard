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

try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# --- [1. 기본 설정 및 다크모드 폰트] ---
st.set_page_config(page_title="Retail Spatial Analytics", layout="wide")

# Altair 및 Matplotlib 다크 테마 적용
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
main_category = st.sidebar.radio("Modules", ["Traffic Summary", "Heatmap Analysis", "AI Operations", "Sensor Map"])

if main_category == "AI Operations":
    st.sidebar.markdown("<hr style='margin: 10px 0; border-color: #334155;'>", unsafe_allow_html=True) 
    sub_menu = st.sidebar.radio("AI Modules", ["Demand Forecast", "Layout Simulator", "LLM Assistant"])
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
                
                # ⭐ 1. OS 비율 전광판
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
                
                # ⭐ 2. 수직선 탑재 고급 트렌드 차트
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
                            x='Time:T',
                            y='Trend:Q',
                            tooltip=[
                                alt.Tooltip('Time:T', format='%H:%M', title='Time'), 
                                alt.Tooltip('visitors:Q', title='Raw Visitors'),
                                alt.Tooltip('Trend:Q', format='.1f', title='Trend (Avg)')
                            ]
                        )
                        
                        peak_point = alt.Chart(pd.DataFrame({'Time': [peak_time], 'visitors': [peak_val]})).mark_circle(
                            size=120, color='#F43F5E', opacity=1
                        ).encode(x='Time:T', y='visitors:Q')
                        
                        peak_text = alt.Chart(pd.DataFrame({'Time': [peak_time], 'visitors': [peak_val]})).mark_text(
                            align='left', baseline='middle', dx=12, dy=-12, color='#F43F5E', fontSize=14, fontWeight='bold',
                            text=f'🔥 Peak: {peak_val:.0f}'
                        ).encode(x='Time:T', y='visitors:Q')
                        
                        final_combo_chart = (area_chart + line_chart + peak_point + peak_text).properties(height=350)
                        st.altair_chart(final_combo_chart, use_container_width=True)
                except Exception as e: 
                    st.error(f"Chart Render Error: {e}")
                
                # ⭐ 3. 드디어! 찐 체류시간 4사분면 차트 (Magic Quadrant)
                st.markdown("<br>#### Zone Performance (Magic Quadrant)", unsafe_allow_html=True)
                with st.spinner("Calculating Dwell Times..."):
                    if 'stay_sec' in filtered_df.columns:
                        zone_stats = filtered_df.groupby('zone').agg(
                            Visitors=('real_user_id', 'nunique'),
                            Avg_Dwell_Time=('stay_sec', lambda x: x.mean() / 60.0) 
                        ).reset_index()
                    else:
                        zone_user_stats = filtered_df.groupby(['zone', 'real_user_id']).size().reset_index(name='log_count')
                        zone_user_stats['dwell_time_min'] = (zone_user_stats['log_count'] * 10) / 60.0 
                        zone_stats = zone_user_stats.groupby('zone').agg(
                            Visitors=('real_user_id', 'nunique'),
                            Avg_Dwell_Time=('dwell_time_min', 'mean')
                        ).reset_index()

                    if not zone_stats.empty:
                        avg_vis = zone_stats['Visitors'].mean()
                        avg_dwell = zone_stats['Avg_Dwell_Time'].mean()
                        
                        scatter = alt.Chart(zone_stats).mark_circle(size=250, opacity=0.8, color='#8B5CF6').encode(
                            x=alt.X('Visitors:Q', title='Unique Visitors (인기도)', scale=alt.Scale(zero=False), axis=alt.Axis(gridColor='#334155', domainColor='#334155')),
                            y=alt.Y('Avg_Dwell_Time:Q', title='Average Dwell Time [Min] (체류시간)', scale=alt.Scale(zero=False), axis=alt.Axis(gridColor='#334155', domainColor='#334155')),
                            tooltip=['zone', 'Visitors', alt.Tooltip('Avg_Dwell_Time:Q', format='.1f', title='Dwell Time (Min)')]
                        )
                        
                        text = scatter.mark_text(
                            align='left', baseline='middle', dx=12, color='#0F172A', fontSize=12, fontWeight=600
                        ).encode(text='zone')
                        
                        hline = alt.Chart(pd.DataFrame({'y': [avg_dwell]})).mark_rule(color='#F43F5E', strokeDash=[4,4], strokeWidth=1.5).encode(y='y:Q')
                        vline = alt.Chart(pd.DataFrame({'x': [avg_vis]})).mark_rule(color='#F43F5E', strokeDash=[4,4], strokeWidth=1.5).encode(x='x:Q')
                        
                        quadrant_chart = (scatter + text + hline + vline).properties(height=450)
                        
                        st.markdown("""
                        <div style="background-color: #1E293B; padding: 10px 15px; border-radius: 8px; border-left: 3px solid #8B5CF6; margin-bottom: 10px;">
                            <span style="color: #94A3B8; font-size: 13px;">
                            💡 <b>해석 방법:</b> 십자선(빨간 점선)은 전체 평균입니다. <br>
                            - <b>우상단:</b> 방문객도 많고 오래 머무는 <b>핵심 매출 구역 (Golden Zone)</b><br>
                            - <b>우하단:</b> 스쳐 지나가는 <b>통로 구역</b> (충동구매 상품 배치 권장)<br>
                            - <b>좌상단:</b> 소수 마니아가 꼼꼼히 고르는 <b>목적 구매 구역</b><br>
                            - <b>좌하단:</b> 방문객도 없고 빨리 나가는 <b>개선 필요 구역 (Dead Zone)</b>
                            </span>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.altair_chart(quadrant_chart, use_container_width=True)
                
                # ⭐ 4. 고객 동선 맵
                st.markdown("<br>#### Customer Flow Map", unsafe_allow_html=True)
                with st.spinner("Rendering flow map..."):
                    flow_df = filtered_df.copy()
                    if 'next_zone' not in flow_df.columns and 'enter_time' in flow_df.columns:
                        flow_df = flow_df.sort_values(['real_user_id', 'enter_time'])
                        flow_df['next_zone'] = flow_df.groupby('real_user_id')['zone'].shift(-1)
                    
                    if 'next_zone' in flow_df.columns:
                        flow_df = flow_df.dropna(subset=['next_zone'])
                        flow_df = flow_df[flow_df['zone'] != flow_df['next_zone']]
                        flow_counts = flow_df.groupby(['zone', 'next_zone']).size().reset_index(name='weight')
                        
                        if not flow_counts.empty:
                            top_flows = flow_counts.sort_values('weight', ascending=False).head(100)
                            zone_popularity = filtered_df['zone'].value_counts().to_dict()
                            G = nx.DiGraph()
                            for zone_name in ZONES.keys(): G.add_node(zone_name)
                            for _, row in top_flows.iterrows(): G.add_edge(row['zone'], row['next_zone'], weight=row['weight'])
                            pos = {node: ((ZONES[node]['x_min']+ZONES[node]['x_max'])/2, (ZONES[node]['y_min']+ZONES[node]['y_max'])/2) if node in ZONES else (331, 250) for node in G.nodes()}
                            
                            fig_flow, ax_flow = plt.subplots(figsize=(12, 9), dpi=150)
                            fig_flow.patch.set_facecolor('white')
                            ax_flow.set_facecolor('white')
                            img_path = 'map_image.jpg'
                            try:
                                img = mpimg.imread(img_path)
                                ax_flow.imshow(img, extent=[0, 663, 500, 0], alpha=0.5)
                            except: ax_flow.set_xlim(0, 663); ax_flow.set_ylim(500, 0); ax_flow.invert_yaxis()
                            
                            max_pop = max(list(zone_popularity.values())) if zone_popularity.values() else 1
                            node_sizes = [(zone_popularity.get(node, 0) / max_pop) * 1500 + 100 for node in G.nodes()]
                            node_colors = ['#FFB347' if zone_popularity.get(node, 0) > 0 else '#B0BEC5' for node in G.nodes()]
                            max_weight = max([G[u][v]['weight'] for u, v in G.edges()]) if G.edges() else 1
                            edge_widths = [(G[u][v]['weight'] / max_weight) * 3 + 0.5 for u, v in G.edges()]
                            
                            nx.draw_networkx_nodes(G, pos, ax=ax_flow, node_size=node_sizes, node_color=node_colors, edgecolors='black', linewidths=1.2, alpha=0.85)
                            nx.draw_networkx_edges(G, pos, ax=ax_flow, width=edge_widths, edge_color='#D84315', arrowsize=15, alpha=0.6, connectionstyle='arc3,rad=0.2')
                            nx.draw_networkx_labels(G, pos, ax=ax_flow, font_family=plt.rcParams['font.family'], font_size=9, font_weight='bold', font_color='black', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
                            ax_flow.axis('off')
                            st.pyplot(fig_flow, facecolor='white')
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
                        
                        highlight = alt.selection_point(fields=['Label'], bind='legend')
                        
                        chart_multi = alt.Chart(plot_data_multi).mark_line(
                            interpolate='monotone', 
                            strokeWidth=3
                        ).encode(
                            x=alt.X('Time:T', title='Time', axis=alt.Axis(format='%H:%M', grid=True, gridColor='#475569', gridDash=[4, 4], gridWidth=0.8, tickCount=15, domainColor='#334155')),
                            y=alt.Y('visitors:Q', title='Visitors', axis=alt.Axis(gridColor='#334155', domainColor='#334155')),
                            color=alt.Color('Label:N', title='Legend (Click to Isolate)', scale=alt.Scale(scheme='set2')),
                            opacity=alt.condition(highlight, alt.value(1.0), alt.value(0.1)),
                            tooltip=['Label:N', alt.Tooltip('Time:T', format='%H:%M'), 'visitors:Q']
                        ).add_params(highlight)
                        
                        st.altair_chart(chart_multi.properties(height=400), use_container_width=True)
                        
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
                except: pass

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
                fig.patch.set_facecolor('white')
                ax.set_facecolor('white')
                
                if os.path.exists('map_image.jpg'): ax.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], zorder=1, alpha=0.5)
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
                    st.pyplot(fig, facecolor='white')

elif menu == "Demand Forecast":
    st.title("Demand Forecast")
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
            
        if st.button("Run Forecast", use_container_width=True):
            try:
                ai_model = joblib.load("ai_forecaster.pkl")
                features = joblib.load("ai_features.pkl")
                target_zones = ['라면', '채소/계란/과일', '주류', '장난감']
                predictions = {}
                for zone in target_zones:
                    input_data = pd.DataFrame(columns=features)
                    input_data.loc[0] = 0 
                    input_data['Is_Weekend'] = is_weekend; input_data['Is_Holiday'] = is_holiday; input_data['Is_Working_Holiday'] = 1 if (is_holiday and not is_weekend) else 0; input_data['Is_Weekend_Holiday'] = 1 if (is_holiday and is_weekend) else 0; input_data['Is_Long_Holiday'] = is_long_holiday; input_data['Is_Pre_Holiday'] = is_pre_holiday; input_data['Is_Post_Holiday'] = is_post_holiday
                    if "Sunny" in future_weather: input_data['Weather_Clean_Sunny'] = 1
                    elif "Cloudy" in future_weather: input_data['Weather_Clean_Cloudy'] = 1
                    elif "Rainy" in future_weather: input_data['Weather_Clean_Rainy'] = 1
                    if f"DayName_Clean_{future_dayname}" in input_data.columns: input_data[f"DayName_Clean_{future_dayname}"] = 1
                    if f"zone_{zone}" in input_data.columns: input_data[f"zone_{zone}"] = 1
                    predictions[zone] = ai_model.predict(input_data)[0]
                
                try:
                    trend_df = pd.read_csv("time_trend_light.csv")
                    hourly_ratio = trend_df.groupby('time_str')['visitors'].sum() / trend_df['visitors'].sum()
                    total_predicted = sum(predictions.values()) * 2.5 
                    pred_curve = (hourly_ratio * total_predicted).reset_index()
                    pred_curve.columns = ['Time', 'Expected Visitors']
                    
                    base_date = pd.to_datetime("2026-01-01")
                    pred_curve['Time'] = pd.to_datetime(base_date.strftime('%Y-%m-%d') + ' ' + pred_curve['Time'])
                    
                    st.markdown("#### Forecasted Traffic Curve")
                    chart = alt.Chart(pred_curve).mark_area(
                        interpolate='monotone', color='#8B5CF6', opacity=0.3
                    ).encode(
                        x=alt.X('Time:T', axis=alt.Axis(format='%H:%M', grid=True, gridColor='#475569', gridDash=[4, 4], gridWidth=0.8, tickCount=15, domainColor='#334155')),
                        y=alt.Y('Expected Visitors:Q', axis=alt.Axis(gridColor='#334155', domainColor='#334155'))
                    ) + alt.Chart(pred_curve).mark_line(
                        interpolate='monotone', color='#A78BFA', strokeWidth=2
                    ).encode(x=alt.X('Time:T'), y=alt.Y('Expected Visitors:Q'))
                    st.altair_chart(chart.properties(height=250), use_container_width=True)
                except: pass

                st.markdown("#### Zone Insights")
                for zone, traffic in predictions.items():
                    insight = ""
                    if is_pre_holiday and zone == '주류': insight = " - [AI Insight] 공휴일 전날 수요 급증. 재고 및 전진 배치 권장."
                    elif is_post_holiday and zone == '채소/계란/과일': insight = " - [AI Insight] 연휴 직후 신선식품 수요 집중. 재고 1.3배 확보 요망."
                    elif "Rainy" in future_weather and zone == '라면': insight = " - [AI Insight] 우천 시 국물 요리 수요 증가."
                    
                    st.markdown(f"- **{zone}**: {traffic:,.0f} visits {insight}")

            except: st.error("Model 'ai_forecaster.pkl' not found.")

elif menu == "Layout Simulator":
    st.title("Layout Simulator")

    if df_all is not None:
        col1, col2 = st.columns(2)
        zone_list = list(ZONES.keys())
        with col1: swap_a = st.selectbox("Target Zone A", zone_list, index=zone_list.index('라면') if '라면' in zone_list else 0)
        with col2: swap_b = st.selectbox("Target Zone B", zone_list, index=zone_list.index('주류') if '주류' in zone_list else 1)

        if swap_a == swap_b: st.warning("Please select distinct zones.")
        else:
            if st.button("Run Simulation", use_container_width=True):
                with st.spinner("Processing layout changes..."):
                    
                    orig_centers = {node: ((ZONES[node]['x_min']+ZONES[node]['x_max'])/2, (ZONES[node]['y_min']+ZONES[node]['y_max'])/2) for node in ZONES}
                    sim_centers = orig_centers.copy()
                    sim_centers[swap_a], sim_centers[swap_b] = orig_centers[swap_b], orig_centers[swap_a]

                    flow_df = df_all.copy()
                    if 'next_zone' not in flow_df.columns:
                        flow_df = flow_df.sort_values(['real_user_id', 'enter_time'])
                        flow_df['next_zone'] = flow_df.groupby('real_user_id')['zone'].shift(-1)
                    flow_df = flow_df.dropna(subset=['next_zone'])
                    flow_df = flow_df[flow_df['zone'] != flow_df['next_zone']]
                    
                    all_flows = flow_df.groupby(['zone', 'next_zone']).size().reset_index(name='weight')
                    sim_flows = all_flows.copy()
                    
                    zone_popularity = df_all['zone'].value_counts().to_dict()
                    sim_zone_pop = zone_popularity.copy()

                    def calc_dist(p1, p2): return math.hypot(p1[0]-p2[0], p1[1]-p2[1])
                    
                    for idx, row in sim_flows.iterrows():
                        u, v = row['zone'], row['next_zone']
                        if u in sim_centers and v in sim_centers:
                            old_d = calc_dist(orig_centers[u], orig_centers[v])
                            new_d = calc_dist(sim_centers[u], sim_centers[v])
                            
                            if old_d != new_d and old_d > 0 and new_d > 0:
                                ratio = max(0.5, min(old_d / new_d, 2.0))
                                new_weight = row['weight'] * ratio
                                diff = new_weight - row['weight']
                                
                                sim_flows.at[idx, 'weight'] = int(new_weight)
                                sim_zone_pop[u] = sim_zone_pop.get(u, 0) + diff
                                sim_zone_pop[v] = sim_zone_pop.get(v, 0) + diff
                    
                    zones_data = []
                    for k in ZONES.keys():
                        orig_val = zone_popularity.get(k, 0)
                        sim_val = int(sim_zone_pop.get(k, 0))
                        diff = sim_val - orig_val
                        zones_data.append({'Zone': k, 'Visits': max(0, sim_val), 'Variance': diff})
                        
                    df_zones_sim = pd.DataFrame(zones_data).sort_values('Visits', ascending=False)

                    def get_color(row):
                        if row['Zone'] in [swap_a, swap_b]: return '#8B5CF6' 
                        elif row['Variance'] > 0: return '#10B981' 
                        elif row['Variance'] < 0: return '#F43F5E' 
                        else: return '#334155' 
                        
                    df_zones_sim['Color'] = df_zones_sim.apply(get_color, axis=1)
                    
                    st.markdown("#### Impact Analysis")
                    sim_a_traffic = int(sim_zone_pop.get(swap_a, 0))
                    sim_b_traffic = int(sim_zone_pop.get(swap_b, 0))
                    diff_a = sim_a_traffic - zone_popularity.get(swap_a, 0)
                    diff_b = sim_b_traffic - zone_popularity.get(swap_b, 0)
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric(f"Zone: {swap_a}", f"{sim_a_traffic:,}", f"{diff_a:,}")
                    metric_col2.metric(f"Zone: {swap_b}", f"{sim_b_traffic:,}", f"{diff_b:,}")
                    
                    other_diffs = {k: v for k, v in zip(df_zones_sim['Zone'], df_zones_sim['Variance']) if k not in [swap_a, swap_b]}
                    top_gainer = max(other_diffs, key=other_diffs.get) if other_diffs else None
                    top_gainer_diff = other_diffs.get(top_gainer, 0) if top_gainer else 0
                    
                    if top_gainer and top_gainer_diff > 0:
                        metric_col3.metric(f"Primary Beneficiary: {top_gainer}", f"{int(sim_zone_pop.get(top_gainer, 0)):,}", f"{top_gainer_diff:,}")
                    else:
                        metric_col3.metric("Spillover Impact", "N/A", "Distance decay")
                    
                    st.markdown("<br>#### Post-Simulation Distribution", unsafe_allow_html=True)
                    bars = alt.Chart(df_zones_sim).mark_bar(cornerRadiusEnd=3, height=15).encode(
                        x=alt.X('Visits:Q', axis=alt.Axis(gridColor='#334155', domainColor='#334155')),
                        y=alt.Y('Zone:N', sort='-x', title='', axis=alt.Axis(gridColor='#334155', domainColor='#334155')),
                        color=alt.Color('Color:N', scale=None, legend=None),
                        tooltip=['Zone', 'Visits', 'Variance']
                    )
                    st.altair_chart(bars.properties(height=alt.Step(22)), use_container_width=True)

                    st.markdown("#### Simulated Flow Graph")
                    top_100_sim_flows = sim_flows.sort_values('weight', ascending=False).head(100).copy()
                    
                    G_sim = nx.DiGraph()
                    for zone_name in ZONES.keys(): G_sim.add_node(zone_name)
                    for _, row in top_100_sim_flows.iterrows(): G_sim.add_edge(row['zone'], row['next_zone'], weight=row['weight'])
                    
                    fig_sim, ax_sim = plt.subplots(figsize=(12, 9), dpi=150)
                    fig_sim.patch.set_facecolor('white')
                    ax_sim.set_facecolor('white')
                    if os.path.exists('map_image.jpg'): ax_sim.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], alpha=0.5)
                    else: ax_sim.set_xlim(0, 663); ax_sim.set_ylim(500, 0); ax_sim.invert_yaxis()
                    
                    max_pop = max(list(sim_zone_pop.values())) if sim_zone_pop.values() else 1
                    
                    node_colors = []
                    for node in G_sim.nodes():
                        diff = int(sim_zone_pop.get(node, 0)) - zone_popularity.get(node, 0)
                        if node in [swap_a, swap_b]: node_colors.append('#8B5CF6')
                        elif diff > 0: node_colors.append('#10B981')
                        elif diff < 0: node_colors.append('#F43F5E')
                        else: node_colors.append('#CBD5E1')
                    
                    node_sizes = [(sim_zone_pop.get(node, 0) / max_pop) * 1500 + 100 for node in G_sim.nodes()]
                    max_weight = max([G_sim[u][v]['weight'] for u, v in G_sim.edges()]) if G_sim.edges() else 1
                    edge_widths = [(G_sim[u][v]['weight'] / max_weight) * 3 + 0.5 for u, v in G_sim.edges()]
                    
                    nx.draw_networkx_nodes(G_sim, sim_centers, ax=ax_sim, node_size=node_sizes, node_color=node_colors, edgecolors='black', linewidths=1.2)
                    nx.draw_networkx_edges(G_sim, sim_centers, ax=ax_sim, width=edge_widths, edge_color='#6366F1', arrowsize=15, alpha=0.6, connectionstyle='arc3,rad=0.2')
                    nx.draw_networkx_labels(G_sim, sim_centers, ax=ax_sim, font_family=plt.rcParams['font.family'], font_size=9, font_weight='bold', font_color='black', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
                    
                    ax_sim.axis('off')
                    st.pyplot(fig_sim, facecolor='white')

elif menu == "LLM Assistant":
    st.title("LLM Operations Advisor")
    if not HAS_GENAI: st.error("google-generativeai module not found.")
    else:
        with st.container():
            try:
                genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
                best_model = 'gemini-1.5-flash'
                try:
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            if 'flash' in m.name: best_model = m.name; break
                            elif 'pro' in m.name: best_model = m.name
                except: pass
                model = genai.GenerativeModel(best_model)
                
                if "chat_history" not in st.session_state: st.session_state.chat_history = []
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])
                if prompt := st.chat_input("Ask advisor..."):
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("Processing..."):
                            response = model.generate_content(prompt)
                            st.markdown(response.text)
                            st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            except KeyError: st.error("API Key not found in st.secrets.")

elif menu == "Sensor Map":
    st.title("Hardware Deployment Map")
    try:
        sward_df = pd.read_csv('swards (1).csv')
        fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
        fig.patch.set_facecolor('white')
        ax.set_facecolor('white')
        if os.path.exists('map_image.jpg'): ax.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], zorder=1, alpha=0.5)
        else: ax.set_xlim(0, 663); ax.set_ylim(500, 0); ax.invert_yaxis()
        ax.scatter(sward_df['x'], sward_df['y'], color='#F43F5E', s=55, edgecolors='black', linewidth=1, zorder=2)
        for _, row in sward_df.iterrows(): ax.annotate(str(row['description']), (row['x'], row['y']), xytext=(5, 5), textcoords='offset points', fontsize=8, color='black', fontweight='bold')
        ax.axis('off')
        st.pyplot(fig, facecolor='white')
    except: st.error("Sensor configuration file not found.")
