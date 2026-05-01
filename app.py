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

# ⭐ [추가됨] 제미나이 인공지능 라이브러리
try:
    import google.generativeai as genai
    HAS_GENAI = True
except ImportError:
    HAS_GENAI = False

# --- [1. 기본 설정 및 한글 폰트] ---
st.set_page_config(page_title="Retail AI Dashboard", page_icon="🛒", layout="wide")

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
    .stApp { background-color: #F8FAFC; }
    [data-testid="stSidebar"] { background-color: #1E293B !important; }
    [data-testid="stSidebar"] * { color: #F1F5F9 !important; }
    h1, h2, h3 { color: #0F172A; font-weight: 800 !important; letter-spacing: -0.5px; }
    [data-testid="stMetric"] { background-color: #FFFFFF; padding: 20px; border-radius: 15px; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0; text-align: center; transition: transform 0.2s; }
    [data-testid="stMetric"]:hover { transform: translateY(-5px); }
    [data-testid="stMetricLabel"] { font-size: 15px; color: #64748B; font-weight: 600; }
    [data-testid="stMetricValue"] { font-size: 36px; color: #2563EB; font-weight: 900; }
    [data-testid="stVerticalBlockBorderWrapper"] { background-color: #FFFFFF !important; border-radius: 15px !important; box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important; border: 1px solid #E2E8F0 !important; padding: 15px !important; }
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
                is_holiday = str(row['is_holiday']).strip().lower() == 'true'
                is_weekend = str(row['is_weekend']).strip().lower() == 'true'
                
                if is_holiday: holiday_text = "🔴 공휴일"
                elif is_weekend: holiday_text = "🟡 휴일(주말)"
                else: holiday_text = "🟢 평일"
                
                weather_lower = weather.lower()
                if "rain" in weather_lower: icon = "🌧️"
                elif "cloud" in weather_lower: icon = "☁️"
                elif "sun" in weather_lower or "clear" in weather_lower: icon = "☀️"
                else: icon = "🌤️"
                weather_dict[day_num] = f"{date_str} [{icon} {weather} | {holiday_text}]"
        except: pass
    return weather_dict

df_all = load_all_sessions()
weather_info = load_weather()

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
    if d == "전체 누적 보기": return d
    try:
        day_num = int(str(d).split('-')[-1])
        return weather_info.get(day_num, str(d))
    except: return str(d)

st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3082/3082011.png", width=100)
st.sidebar.title("마트 AI 대시보드")
main_category = st.sidebar.radio("📌 메인 메뉴", ["📊 트래픽 요약", "🔥 정밀 히트맵", "🤖 AI 어드바이저", "📍 센서(Sward) 위치"])

if main_category == "🤖 AI 어드바이저":
    st.sidebar.markdown("<hr style='margin: 10px 0; border-color: #334155;'>", unsafe_allow_html=True) 
    sub_menu = st.sidebar.radio("💡 상세 기능 선택", ["🌤️ 내일의 AI 예측 브리핑", "🔄 매대 이동 시뮬레이터", "💬 Gemini 매장 비서 (챗봇)"])
    menu = sub_menu 
else:
    menu = main_category

if menu == "📊 트래픽 요약":
    st.title("📊 마트 트래픽 요약")
    st.markdown("""
    <div style="background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); padding: 30px; border-radius: 15px; border-left: 5px solid #3B82F6; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <h4 style="color: #1E3A8A; margin-top: 0;">🔴 실시간 매장 트래픽 모니터링 (BETA)</h4>
        <p style="color: #475569; font-size: 15px; margin-bottom: 0;">🚧 현재 개발 중인 기능입니다.</p>
    </div>
    """, unsafe_allow_html=True)
    if df_all is not None and 'date' in df_all.columns:
        available_dates = sorted(df_all['date'].unique().tolist(), key=sort_date_smart)
        selected_date = st.selectbox("📅 조회할 날짜를 선택하세요:", ["전체 누적 보기"] + available_dates, format_func=format_date_option)
        if selected_date == "전체 누적 보기":
            filtered_df = df_all
        else:
            filtered_df = df_all[df_all['date'].apply(lambda x: safe_date_match(x, selected_date))]
            
        if not filtered_df.empty:
            total_users = df_all.groupby('date')['real_user_id'].nunique().sum() if selected_date == "전체 누적 보기" else filtered_df['real_user_id'].nunique()
            col1, col2, col3 = st.columns(3)
            total_stays = filtered_df['stay_sec'].sum() / 3600
            top_zone = filtered_df['zone'].value_counts().index[0]
            col1.metric("해당 기간 방문 고객 (연인원)", f"{total_users:,.0f} 명")
            col2.metric("고객 총 체류시간", f"{total_stays:,.0f} 시간")
            col3.metric("가장 붐빈 코너 1위", top_zone)
            
            st.markdown("<br>### 🌊 시간대별 매장 정밀 트래픽 흐름 (10분 단위)", unsafe_allow_html=True)
            try:
                trend_df = pd.read_csv("time_trend_light.csv")
                if selected_date == "전체 누적 보기":
                    plot_data = trend_df.groupby('time_str')['visitors'].sum().reset_index()
                    y_title = '총 누적 방문객 수 (명)'
                else:
                    plot_data = trend_df[trend_df['date'].apply(lambda x: safe_date_match(x, selected_date))]
                    y_title = '동시 체류 방문객 수 (명)'
                if not plot_data.empty:
                    base_date = pd.to_datetime("2026-01-01")
                    plot_data['시간'] = pd.to_datetime(base_date.strftime('%Y-%m-%d') + ' ' + plot_data['time_str'])
                    chart = alt.Chart(plot_data).mark_area(interpolate='monotone', color='#93C5FD', opacity=0.4).encode(
                        x=alt.X('시간:T', title='시간', axis=alt.Axis(format='%H:%M', labelColor='#475569')),
                        y=alt.Y('visitors:Q', title=y_title, axis=alt.Axis(labelColor='#475569')),
                        tooltip=[alt.Tooltip('시간:T', format='%H:%M', title='시간대'), alt.Tooltip('visitors:Q', title='방문객 수')]
                    ) + alt.Chart(plot_data).mark_line(interpolate='monotone', color='#3B82F6', strokeWidth=3).encode(x=alt.X('시간:T'), y=alt.Y('visitors:Q'))
                    st.altair_chart(chart.properties(height=380).interactive(), use_container_width=True)
                else: st.info("💡 선택하신 날짜의 시간대별 트래픽 데이터가 없습니다.")
            except: st.error("그래프 생성 중 오류가 발생했습니다.")
            
            st.markdown("<br>### 🏆 구역별 전체 방문 횟수", unsafe_allow_html=True)
            df_zones = filtered_df['zone'].value_counts().reset_index()
            df_zones.columns = ['구역', '방문횟수']
            bars = alt.Chart(df_zones).mark_bar(cornerRadiusEnd=5).encode(
                x=alt.X('방문횟수:Q', title='방문 횟수 (회)'),
                y=alt.Y('구역:N', sort='-x', title=''),
                color=alt.Color('방문횟수:Q', scale=alt.Scale(scheme='blues'), legend=None),
                tooltip=['구역', '방문횟수']
            )
            text = bars.mark_text(align='left', baseline='middle', dx=5, fontSize=13, fontWeight='bold', color='#1E293B').encode(text=alt.Text('방문횟수:Q', format=','))
            st.altair_chart((bars + text).properties(height=alt.Step(35)), use_container_width=True)
            
            st.markdown("<br>### 🕸️ 매장 내 고객 이동 동선 흐름도 (Flow Map)", unsafe_allow_html=True)
            with st.spinner("동선 흐름도를 렌더링 중입니다..."):
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
                        
                        # ⭐ 수정됨: facecolor='white' 추가하여 배경을 강제로 하얗게 만듭니다!
                        fig_flow, ax_flow = plt.subplots(figsize=(12, 9), dpi=150, facecolor='white')
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
                        nx.draw_networkx_labels(G, pos, ax=ax_flow, font_family=plt.rcParams['font.family'], font_size=9, font_weight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
                        ax_flow.axis('off')
                        st.pyplot(fig_flow)
        else: st.info("데이터가 없습니다.")

elif menu == "🔥 정밀 히트맵":
    st.title("🔥 오리지널 구름 히트맵")
    if df_all is not None and 'date' in df_all.columns:
        available_dates = sorted(df_all['date'].unique().tolist(), key=sort_date_smart)
        selected_date = st.selectbox("📅 조회할 날짜를 선택하세요:", available_dates, key="heatmap_date", format_func=format_date_option)
        target_files = glob.glob(f"*{selected_date}*")
        traj_files = [f for f in target_files if 'trajectory' in f.lower() or 'real_users_trajectory' in f.lower()]
        filtered_traj = pd.DataFrame()
        if traj_files:
            try: filtered_traj = pd.read_parquet(traj_files[0]) if traj_files[0].endswith('.parquet') else pd.read_csv(traj_files[0])
            except: pass
                
        if not filtered_traj.empty:
            col1, col2 = st.columns([1, 3])
            with col1:
                selected_time = st.slider("⏰ 특정 시간 스냅샷", datetime.time(9, 0), datetime.time(22, 50), datetime.time(15, 0), step=datetime.timedelta(minutes=10), format="HH:mm")
                blur_sigma = st.slider("구름 퍼짐 정도", 1.0, 10.0, 4.0, step=0.5)
                red_sens = st.slider("붉은색 민감도", 1, 50, 15, step=1)
            with col2:
                fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
                if os.path.exists('map_image.jpg'): ax.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], zorder=1)
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
                    st.pyplot(fig)
                else: st.warning("데이터가 없습니다.")
        else: st.info(f"⚠️ {selected_date}의 동선 데이터 파일이 없습니다.")

elif menu == "🌤️ 내일의 AI 예측 브리핑":
    st.title("🌤️ 내일의 트래픽 예측 및 AI 브리핑")
    with st.container(border=True):
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1: future_weather = st.selectbox("⛅ 1. 예상 날씨", ["Sunny (맑음)", "Cloudy (흐림)", "Rainy (비/눈)"])
        with row1_col2: future_dayname = st.selectbox("📅 2. 요일 선택", ["Monday (월)", "Tuesday (화)", "Wednesday (수)", "Thursday (목)", "Friday (금)", "Saturday (토)", "Sunday (일)"])
        is_weekend = 1 if "Saturday" in future_dayname or "Sunday" in future_dayname else 0
        weekend_text = "주말" if is_weekend else "평일"

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1: future_holiday = st.selectbox(f"🎈 3. 공휴일 여부", ["No (공휴일 아님)", "Yes (공휴일 맞음)"]); is_holiday = 1 if "Yes" in future_holiday else 0
        is_long_holiday = 0
        if is_holiday:
            with row2_col2: long_holiday_str = st.selectbox("🎒 4. 명절/긴 연휴", ["일반 공휴일 (Short)", "긴 연휴 (Long)"]); is_long_holiday = 1 if "Long" in long_holiday_str else 0
        with row2_col3: pre_post_str = st.selectbox("⏳ 5. 전/후 여부", ["해당 없음", "공휴일 전날", "공휴일 다음날"]); is_pre_holiday = 1 if "Pre" in pre_post_str else 0; is_post_holiday = 1 if "Post" in pre_post_str else 0
            
        if st.button("🤖 AI 예측 및 그래프 생성하기", use_container_width=True):
            try:
                ai_model = joblib.load("ai_forecaster.pkl")
                features = joblib.load("ai_features.pkl")
                target_zones = ['라면', '채소/계란/과일', '주류', '장난감']
                predictions = {}
                day_map = {"Monday (월)": "Monday", "Tuesday (화)": "Tuesday", "Wednesday (수)": "Wednesday", "Thursday (목)": "Thursday", "Friday (금)": "Friday", "Saturday (토)": "Saturday", "Sunday (일)": "Sunday"}
                for zone in target_zones:
                    input_data = pd.DataFrame(columns=features)
                    input_data.loc[0] = 0 
                    input_data['Is_Weekend'] = is_weekend; input_data['Is_Holiday'] = is_holiday; input_data['Is_Working_Holiday'] = 1 if (is_holiday and not is_weekend) else 0; input_data['Is_Weekend_Holiday'] = 1 if (is_holiday and is_weekend) else 0; input_data['Is_Long_Holiday'] = is_long_holiday; input_data['Is_Pre_Holiday'] = is_pre_holiday; input_data['Is_Post_Holiday'] = is_post_holiday
                    if "Sunny" in future_weather: input_data['Weather_Clean_Sunny'] = 1
                    elif "Cloudy" in future_weather: input_data['Weather_Clean_Cloudy'] = 1
                    elif "Rainy" in future_weather: input_data['Weather_Clean_Rainy'] = 1
                    selected_day = day_map[future_dayname]
                    if f"DayName_Clean_{selected_day}" in input_data.columns: input_data[f"DayName_Clean_{selected_day}"] = 1
                    if f"zone_{zone}" in input_data.columns: input_data[f"zone_{zone}"] = 1
                    predictions[zone] = ai_model.predict(input_data)[0]
                st.success("AI 분석 완료! 아래 예측 브리핑을 확인하세요.")
                
                try:
                    trend_df = pd.read_csv("time_trend_light.csv")
                    hourly_ratio = trend_df.groupby('time_str')['visitors'].sum() / trend_df['visitors'].sum()
                    total_predicted = sum(predictions.values()) * 2.5 
                    pred_curve = (hourly_ratio * total_predicted).reset_index()
                    pred_curve.columns = ['시간', '예상방문객']
                    
                    base_date = pd.to_datetime("2026-01-01")
                    pred_curve['시간'] = pd.to_datetime(base_date.strftime('%Y-%m-%d') + ' ' + pred_curve['시간'])
                    
                    st.markdown("### 📈 내일의 예상 시간대별 트래픽 곡선")
                    chart = alt.Chart(pred_curve).mark_area(
                        interpolate='monotone', color='#8B5CF6', opacity=0.3
                    ).encode(
                        x=alt.X('시간:T', title='시간', axis=alt.Axis(format='%H:%M', labelColor='#475569')),
                        y=alt.Y('예상방문객:Q', title='매장 예상 동시 체류객 (명)', axis=alt.Axis(labelColor='#475569')),
                        tooltip=[alt.Tooltip('시간:T', format='%H:%M', title='시간'), alt.Tooltip('예상방문객:Q', format=',.0f', title='예상 방문객')]
                    ) + alt.Chart(pred_curve).mark_line(
                        interpolate='monotone', color='#6D28D9', strokeWidth=3
                    ).encode(x=alt.X('시간:T'), y=alt.Y('예상방문객:Q'))
                    st.altair_chart(chart.properties(height=250), use_container_width=True)
                except Exception as e: pass

                st.markdown("""
                <div style="background-color: #F8FAFC; padding: 25px; border-radius: 15px; border-left: 5px solid #8B5CF6; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                    <h3 style="color: #4C1D95; margin-top: 0;">📋 AI 매장 운영 브리핑</h3>
                """, unsafe_allow_html=True)
                
                holiday_txt = f"(공휴일[{weekend_text}] / {'명절 연휴' if is_long_holiday else '일반'})" if is_holiday else f"({weekend_text})"
                pre_post_txt = " 📌 **[공휴일 전날 효과 적용됨]**" if is_pre_holiday else " 📌 **[공휴일 다음날 효과 적용됨]**" if is_post_holiday else ""
                day_str = future_dayname.split()[1].replace('(', '').replace(')', '') 
                st.markdown(f"**상황 요약:** 내일은 **{future_weather.split()[0]}** 날씨의 **{day_str}요일 {holiday_txt}** 입니다.{pre_post_txt}<br><br>", unsafe_allow_html=True)
                
                for zone, traffic in predictions.items():
                    if is_pre_holiday and zone == '주류':
                        st.markdown(f"🍷 **[{zone}] 코너 예상 방문객: <span style='color:red; font-size:20px;'>{traffic:,.0f}명</span>**", unsafe_allow_html=True)
                        st.markdown(f"👉 **AI 인사이트:** 내일은 공휴일 전날입니다! 퇴근 후 홈파티 수요로 주류/안주류 트래픽이 폭발할 예정입니다. 맥주와 소주 매대를 전진 배치하세요.<br><br>", unsafe_allow_html=True)
                    elif is_post_holiday and zone == '채소/계란/과일':
                        st.markdown(f"🥬 **[{zone}] 코너 예상 방문객: <span style='color:red; font-size:20px;'>{traffic:,.0f}명</span>**", unsafe_allow_html=True)
                        st.markdown(f"👉 **AI 인사이트:** 연휴/공휴일 직후 텅 빈 냉장고를 채우려는 신선식품 수요가 급증합니다. 신선 코너 재고를 평소의 1.3배 이상 확보하세요.<br><br>", unsafe_allow_html=True)
                    elif "Rainy" in future_weather and zone == '라면':
                        st.markdown(f"🍜 **[{zone}] 코너 예상 방문객: <span style='color:red; font-size:20px;'>{traffic:,.0f}명</span>**", unsafe_allow_html=True)
                        st.markdown(f"👉 **AI 인사이트:** 비 오는 날 국물 요리 수요 급증이 예상됩니다. 라면 재고를 보충하세요.<br><br>", unsafe_allow_html=True)
                    elif (is_holiday or is_weekend) and zone == '장난감':
                        st.markdown(f"🧸 **[{zone}] 코너 예상 방문객: <span style='color:red; font-size:20px;'>{traffic:,.0f}명</span>**", unsafe_allow_html=True)
                        st.markdown(f"👉 **AI 인사이트:** 주말/휴일 가족 단위 방문객 증가로 트래픽 폭발이 예상됩니다. 전담 직원을 배치하세요.<br><br>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"🛒 **[{zone}] 코너 예상 방문객: {traffic:,.0f}명**<br><br>", unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)

            except: st.error("AI 모델을 불러오지 못했습니다.")

# ====================================================================
# ⭐ [업그레이드] 매대 이동 시뮬레이터 (모든 구역 색상/숫자 시각화!)
# ====================================================================
elif menu == "🔄 매대 이동 시뮬레이터":
    st.title("🔄 디지털 트윈: 매대 이동 시뮬레이터")
    st.markdown("""
    실제 매대를 물리적으로 옮기기 전에, 대시보드 위에서 두 구역의 위치를 맞바꾸어 봅니다.
    AI(거리-마찰 알고리즘)가 마트 전체의 변경된 동선 거리를 순식간에 재계산하여, 
    **모든 매대의 트래픽 증감(나비효과)을 색상과 정확한 숫자로 한눈에 보여줍니다.**
    """)

    if df_all is not None:
        col1, col2 = st.columns(2)
        zone_list = list(ZONES.keys())
        with col1: swap_a = st.selectbox("🔀 A 매대 선택 (이동할 대상)", zone_list, index=zone_list.index('라면') if '라면' in zone_list else 0)
        with col2: swap_b = st.selectbox("🔀 B 매대 선택 (바뀔 위치)", zone_list, index=zone_list.index('주류') if '주류' in zone_list else 1)

        if swap_a == swap_b: st.warning("서로 다른 두 매대를 선택해주세요.")
        else:
            if st.button("🚀 스와프 시뮬레이션 실행!", use_container_width=True):
                with st.spinner(f"[{swap_a}]와 [{swap_b}]의 위치를 바꾸어 마트 전체의 나비효과를 계산합니다..."):
                    
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
                    
                    # ⭐ 1. 모든 구역의 변동량(diff) 계산하여 데이터프레임 만들기
                    zones_data = []
                    for k in ZONES.keys():
                        orig_val = zone_popularity.get(k, 0)
                        sim_val = int(sim_zone_pop.get(k, 0))
                        diff = sim_val - orig_val
                        zones_data.append({'구역': k, '방문횟수': max(0, sim_val), '변동량': diff})
                        
                    df_zones_sim = pd.DataFrame(zones_data)
                    df_zones_sim = df_zones_sim.sort_values('방문횟수', ascending=False)

                    # ⭐ 2. 색상 규칙 적용 (모든 구역을 증감에 따라 색칠!)
                    def get_color(row):
                        if row['구역'] in [swap_a, swap_b]: return '#8B5CF6' # 보라색 (위치 이동된 매대)
                        elif row['변동량'] > 0: return '#10B981' # 초록색 (수혜 구역)
                        elif row['변동량'] < 0: return '#EF4444' # 빨간색 (피해 구역)
                        else: return '#CBD5E1' # 회색 (변화 없음)
                        
                    df_zones_sim['색상'] = df_zones_sim.apply(get_color, axis=1)
                    
                    # 막대그래프에 붙일 예쁜 텍스트 라벨 (예: 45,000 (▲ 300))
                    df_zones_sim['증감텍스트'] = df_zones_sim['변동량'].apply(lambda x: f"▲ {x:,}" if x > 0 else (f"▼ {abs(x):,}" if x < 0 else "-"))
                    df_zones_sim['라벨'] = df_zones_sim.apply(lambda x: f"{x['방문횟수']:,} ({x['증감텍스트']})", axis=1)

                    st.markdown("### 📊 시뮬레이션 결과 요약 (Before & After)")
                    
                    sim_a_traffic = int(sim_zone_pop.get(swap_a, 0))
                    sim_b_traffic = int(sim_zone_pop.get(swap_b, 0))
                    diff_a = sim_a_traffic - zone_popularity.get(swap_a, 0)
                    diff_b = sim_b_traffic - zone_popularity.get(swap_b, 0)
                    
                    metric_col1, metric_col2, metric_col3 = st.columns(3)
                    metric_col1.metric(f"🔀 [{swap_a}] 매대 변동", f"{sim_a_traffic:,} 회", f"{diff_a:,} 회")
                    metric_col2.metric(f"🔀 [{swap_b}] 매대 변동", f"{sim_b_traffic:,} 회", f"{diff_b:,} 회")
                    
                    other_diffs = {k: v for k, v in zip(df_zones_sim['구역'], df_zones_sim['변동량']) if k not in [swap_a, swap_b]}
                    top_gainer = max(other_diffs, key=other_diffs.get) if other_diffs else None
                    top_gainer_diff = other_diffs.get(top_gainer, 0) if top_gainer else 0
                    
                    if top_gainer and top_gainer_diff > 0:
                        metric_col3.metric(f"📈 최대 수혜: [{top_gainer}]", f"{int(sim_zone_pop.get(top_gainer, 0)):,} 회", f"{top_gainer_diff:,} 회")
                    else:
                        metric_col3.metric(f"📉 주변부 전반적 하락", "감소", "거리 멀어짐")
                    
                    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

                    # ⭐ 3. 전체 구역 방문 횟수 막대그래프 출력
                    st.markdown("### 🏆 전체 구역 예상 방문 횟수 (나비효과 컬러 맵핑)", unsafe_allow_html=True)
                    st.markdown("🟣 **위치 이동** | 🟢 **방문객 증가** | 🔴 **방문객 감소** | ⚪ **변화 없음**")

                    bars = alt.Chart(df_zones_sim).mark_bar(cornerRadiusEnd=5).encode(
                        x=alt.X('방문횟수:Q', title='예상 방문 횟수 (회)'),
                        y=alt.Y('구역:N', sort='-x', title=''),
                        color=alt.Color('색상:N', scale=None, legend=None),
                        tooltip=['구역', '방문횟수', '변동량']
                    )
                    text = bars.mark_text(align='left', baseline='middle', dx=5, fontSize=12, fontWeight='bold', color='#1E293B').encode(text='라벨:N')
                    st.altair_chart((bars + text).properties(height=alt.Step(35)), use_container_width=True)

                    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)

                    # ⭐ 4. 시뮬레이션 네트워크 그래프 (노드 색상도 막대그래프와 통일!)
                    st.markdown("### 🕸️ 시뮬레이션 동선 흐름도 (Flow Map)")
                    top_100_sim_flows = sim_flows.sort_values('weight', ascending=False).head(100).copy()
                    
                    G_sim = nx.DiGraph()
                    for zone_name in ZONES.keys(): G_sim.add_node(zone_name)
                    for _, row in top_100_sim_flows.iterrows(): G_sim.add_edge(row['zone'], row['next_zone'], weight=row['weight'])
                    
                    # ⭐ 수정됨: facecolor='white' 추가하여 배경을 강제로 하얗게 만듭니다!
                    fig_sim, ax_sim = plt.subplots(figsize=(12, 9), dpi=150, facecolor='white')
                    if os.path.exists('map_image.jpg'): ax_sim.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], alpha=0.5)
                    else: ax_sim.set_xlim(0, 663); ax_sim.set_ylim(500, 0); ax_sim.invert_yaxis()
                    
                    max_pop = max(list(sim_zone_pop.values())) if sim_zone_pop.values() else 1
                    
                    node_colors = []
                    for node in G_sim.nodes():
                        diff = int(sim_zone_pop.get(node, 0)) - zone_popularity.get(node, 0)
                        if node in [swap_a, swap_b]: node_colors.append('#8B5CF6') # 보라색
                        elif diff > 0: node_colors.append('#10B981') # 초록색
                        elif diff < 0: node_colors.append('#EF4444') # 빨간색
                        else: node_colors.append('#CBD5E1') # 회색
                    
                    node_sizes = [(sim_zone_pop.get(node, 0) / max_pop) * 1500 + 100 for node in G_sim.nodes()]
                    max_weight = max([G_sim[u][v]['weight'] for u, v in G_sim.edges()]) if G_sim.edges() else 1
                    edge_widths = [(G_sim[u][v]['weight'] / max_weight) * 3 + 0.5 for u, v in G_sim.edges()]
                    
                    nx.draw_networkx_nodes(G_sim, sim_centers, ax=ax_sim, node_size=node_sizes, node_color=node_colors, edgecolors='black', linewidths=1.2, alpha=0.85)
                    nx.draw_networkx_edges(G_sim, sim_centers, ax=ax_sim, width=edge_widths, edge_color='#6366F1', arrowsize=15, alpha=0.6, connectionstyle='arc3,rad=0.2')
                    nx.draw_networkx_labels(G_sim, sim_centers, ax=ax_sim, font_family=plt.rcParams['font.family'], font_size=9, font_weight='bold', bbox=dict(facecolor='white', alpha=0.9, edgecolor='none', boxstyle='round,pad=0.3'))
                    
                    ax_sim.axis('off')
                    st.pyplot(fig_sim)
                    st.success(f"✨ 시뮬레이션 완료! 보라색으로 칠해진 [{swap_a}]와 [{swap_b}] 매대의 위치가 바뀌면서, 마트 내 30개 모든 구역의 트래픽이 연쇄적으로 재계산되었습니다.")

elif menu == "💬 Gemini 매장 비서 (챗봇)":
    st.title("💬 Gemini 매장 운영 비서")
    if not HAS_GENAI: st.error("google-generativeai 라이브러리가 없습니다.")
    else:
        with st.container(border=True):
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
                st.success(f"✅ 구글 서버 자동 연결 성공! (작동 모델: `{best_model}`)")
                
                if "chat_history" not in st.session_state: st.session_state.chat_history = []
                for msg in st.session_state.chat_history:
                    with st.chat_message(msg["role"]): st.markdown(msg["content"])
                if prompt := st.chat_input("질문을 입력하세요..."):
                    st.session_state.chat_history.append({"role": "user", "content": prompt})
                    with st.chat_message("user"): st.markdown(prompt)
                    with st.chat_message("assistant"):
                        with st.spinner("분석 중..."):
                            response = model.generate_content("대형 마트 점장 비서로서 답변: " + prompt)
                            st.markdown(response.text)
                            st.session_state.chat_history.append({"role": "assistant", "content": response.text})
            except KeyError: st.error("비밀 금고에 API 키가 없습니다!")

elif menu == "📍 센서(Sward) 위치":
    st.title("📍 매장 내 센서(Sward) 설치 위치")
    try:
        sward_df = pd.read_csv('swards (1).csv')
        fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
        if os.path.exists('map_image.jpg'): ax.imshow(mpimg.imread('map_image.jpg'), extent=[0, 663, 500, 0], zorder=1)
        else: ax.set_xlim(0, 663); ax.set_ylim(500, 0); ax.invert_yaxis()
        ax.scatter(sward_df['x'], sward_df['y'], color='#EF4444', s=55, edgecolors='white', linewidth=2, zorder=2)
        for _, row in sward_df.iterrows(): ax.annotate(str(row['description']), (row['x'], row['y']), xytext=(5, 5), textcoords='offset points', fontsize=8)
        ax.axis('off')
        st.pyplot(fig)
    except: st.error("'swards (1).csv' 파일을 찾을 수 없습니다.")
