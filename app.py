import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import os
import platform
import re
import glob
import altair as alt
import datetime
import joblib

# --- [1. 기본 설정 및 한글 폰트] ---
st.set_page_config(page_title="Retail AI Dashboard", page_icon="🛒", layout="wide")

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
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
def load_trajectory():
    files = glob.glob("Real_Users_Trajectory*.*") + glob.glob("trajectory_*.*")
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
                
                if is_holiday:
                    holiday_text = "🔴 공휴일"
                elif is_weekend:
                    holiday_text = "🟡 휴일(주말)"
                else:
                    holiday_text = "🟢 평일"
                
                weather_lower = weather.lower()
                if "rain" in weather_lower: icon = "🌧️"
                elif "cloud" in weather_lower: icon = "☁️"
                elif "sun" in weather_lower or "clear" in weather_lower: icon = "☀️"
                else: icon = "🌤️"
                
                weather_dict[day_num] = f"{date_str} [{icon} {weather} | {holiday_text}]"
        except: pass
    return weather_dict

df_all = load_all_sessions()
df_traj = load_trajectory()
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

# ⭐ 시뮬레이터 메뉴가 안정성을 위해 제거되었습니다.
if main_category == "🤖 AI 어드바이저":
    st.sidebar.markdown("<hr style='margin: 10px 0; border-color: #334155;'>", unsafe_allow_html=True) 
    sub_menu = st.sidebar.radio("💡 상세 기능 선택", ["🌤️ 내일의 AI 예측 브리핑"])
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
            st.markdown("### 📈 전체 누적 트래픽")
            total_users = df_all.groupby('date')['real_user_id'].nunique().sum()
        else:
            filtered_df = df_all[df_all['date'].apply(lambda x: safe_date_match(x, selected_date))]
            display_title = format_date_option(selected_date)
            st.markdown(f"### 📈 {display_title} 트래픽")
            total_users = filtered_df['real_user_id'].nunique()
            
        if not filtered_df.empty:
            col1, col2, col3 = st.columns(3)
            total_stays = filtered_df['stay_sec'].sum() / 3600
            top_zone = filtered_df['zone'].value_counts().index[0]
            col1.metric("해당 기간 방문 고객 (연인원)", f"{total_users:,.0f} 명")
            col2.metric("고객 총 체류시간", f"{total_stays:,.0f} 시간")
            col3.metric("가장 붐빈 코너 1위", top_zone)
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🌊 시간대별 매장 정밀 트래픽 흐름 (10분 단위)")
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
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("### 🏆 구역별 전체 방문 횟수")
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
        else: st.info("데이터가 없습니다.")
    else: st.error("데이터 파일에 날짜 정보가 없습니다. 깃허브에 데이터 파일이 존재하는지 확인해주세요.")

elif menu == "🔥 정밀 히트맵":
    st.title("🔥 오리지널 구름 히트맵")
    st.markdown("특정 시간을 선택하여 그 순간 사람들의 동선이 어떻게 분포되어 있는지 **스냅샷**으로 확인하세요.")
    if df_traj is not None and 'date' in df_traj.columns:
        available_dates = sorted(df_traj['date'].unique().tolist(), key=sort_date_smart)
        selected_date = st.selectbox("📅 조회할 날짜를 선택하세요:", ["전체 누적 보기"] + available_dates, key="heatmap_date", format_func=format_date_option)
        if selected_date == "전체 누적 보기":
            filtered_traj = df_traj
            st.markdown("### 📈 전체 누적 동선 히트맵")
        else:
            filtered_traj = df_traj[df_traj['date'].apply(lambda x: safe_date_match(x, selected_date))]
            display_title = format_date_option(selected_date)
            st.markdown(f"### 📈 {display_title} 동선 히트맵")
        if not filtered_traj.empty:
            col1, col2 = st.columns([1, 3])
            with col1:
                with st.container(border=True):
                    st.markdown("<h4 style='color: #1E293B; margin-top:0; font-size:18px;'>🎛️ 히트맵 컨트롤러</h4>", unsafe_allow_html=True)
                    selected_time = st.slider("⏰ 특정 시간 스냅샷 보기", min_value=datetime.time(9, 0), max_value=datetime.time(22, 50), value=datetime.time(15, 0), step=datetime.timedelta(minutes=10), format="HH:mm")
                    end_time = (datetime.datetime.combine(datetime.date.today(), selected_time) + datetime.timedelta(minutes=10)).time()
                    st.markdown(f"<p style='color:#2563EB; font-weight:bold; font-size:14px; text-align:center; background-color:#EFF6FF; padding:5px; border-radius:5px;'>📸 찰칵! [{selected_time.strftime('%H:%M')} ~ {end_time.strftime('%H:%M')}]</p>", unsafe_allow_html=True)
                    st.markdown("<hr style='margin:10px 0;'>", unsafe_allow_html=True)
                    blur_sigma = st.slider("구름 퍼짐 정도 (Sigma)", 1.0, 10.0, 4.0, step=0.5)
                    red_sens = st.slider("붉은색 민감도 (%)", 1, 50, 15, step=1)
            with col2:
                fig, ax = plt.subplots(figsize=(10, 7), dpi=100)
                img_path = 'map_image.jpg'
                if os.path.exists(img_path):
                    img = mpimg.imread(img_path)
                    ax.imshow(img, extent=[0, 663, 500, 0], zorder=1)
                else:
                    ax.set_xlim(0, 663); ax.set_ylim(500, 0); ax.invert_yaxis()
                df_exact = filtered_traj[(filtered_traj['x'] >= 0) & (filtered_traj['x'] <= 663) & (filtered_traj['y'] >= 0) & (filtered_traj['y'] <= 500)].copy()
                if 'time_index' in df_exact.columns and not df_exact.empty:
                    time_idx = pd.to_numeric(df_exact['time_index'], errors='coerce').fillna(0)
                    total_secs = (time_idx * 10) % 86400
                    target_sec = selected_time.hour * 3600 + selected_time.minute * 60
                    df_exact = df_exact[(total_secs >= target_sec) & (total_secs < target_sec + 600)]
                if len(df_exact) > 0:
                    heatmap_grid, _, _ = np.histogram2d(df_exact['y'], df_exact['x'], bins=[100, 132], range=[[0, 500], [0, 663]])
                    heatmap_smoothed = gaussian_filter(heatmap_grid, sigma=blur_sigma)
                    max_val = np.max(heatmap_smoothed)
                    if max_val > 0:
                        red_threshold = max_val * (red_sens / 100.0)
                        vmin_level = max_val * 0.01
                        im = ax.imshow(heatmap_smoothed, extent=[0, 663, 500, 0], cmap='Reds', alpha=0.6, zorder=3, interpolation='bilinear', vmin=vmin_level, vmax=red_threshold)
                    ax.axis('off')
                    st.pyplot(fig)
                else: st.warning("⚠️ 선택하신 스냅샷 시간대에는 고객 동선 데이터가 없습니다.")
        else: st.info("선택한 날짜에 동선 데이터가 없습니다.")
    else: st.error("데이터에 날짜 정보가 없거나 궤적(Trajectory) 파일이 없습니다.")

elif menu == "🌤️ 내일의 AI 예측 브리핑":
    st.title("🌤️ 내일의 트래픽 예측 및 AI 브리핑")
    st.markdown("머신러닝이 **날씨, 요일별 평일/주말 판별, 공휴일 여부(전/후 포함)**를 종합 분석하여 운영 방안을 제안합니다.")
    
    with st.container(border=True):
        st.markdown("<h4 style='color: #1E293B; margin-top:0;'>🔮 내일의 상황을 입력해주세요 (자동계산 지원)</h4>", unsafe_allow_html=True)
        
        row1_col1, row1_col2 = st.columns(2)
        with row1_col1:
            future_weather = st.selectbox("⛅ 1. 예상 날씨", ["Sunny (맑음)", "Cloudy (흐림)", "Rainy (비/눈)"])
        with row1_col2:
            future_dayname = st.selectbox("📅 2. 요일 선택 (평일/주말 자동분류)", 
                                          ["Monday (월)", "Tuesday (화)", "Wednesday (수)", "Thursday (목)", 
                                           "Friday (금)", "Saturday (토)", "Sunday (일)"])
            
        is_weekend = 1 if "Saturday" in future_dayname or "Sunday" in future_dayname else 0
        weekend_text = "주말" if is_weekend else "평일"

        row2_col1, row2_col2, row2_col3 = st.columns(3)
        with row2_col1:
            future_holiday = st.selectbox(f"🎈 3. 공휴일 여부 (현재 {weekend_text})", ["No (공휴일 아님)", "Yes (공휴일 맞음)"])
            is_holiday = 1 if "Yes" in future_holiday else 0
            
        is_long_holiday = 0
        if is_holiday:
            with row2_col2:
                long_holiday_str = st.selectbox("🎒 4. 명절/긴 연휴 여부", ["일반적인 공휴일 (Short)", "설/추석 등 긴 연휴 (Long)"])
                is_long_holiday = 1 if "Long" in long_holiday_str else 0
                
        with row2_col3:
            pre_post_str = st.selectbox("⏳ 5. 공휴일 전/후 여부", ["해당 없음 (일반적인 날)", "공휴일 전날 (Pre-Holiday)", "공휴일 다음날 (Post-Holiday)"])
            is_pre_holiday = 1 if "Pre" in pre_post_str else 0
            is_post_holiday = 1 if "Post" in pre_post_str else 0
            
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
                    
                    input_data['Is_Weekend'] = is_weekend
                    input_data['Is_Holiday'] = is_holiday
                    input_data['Is_Working_Holiday'] = 1 if (is_holiday == 1 and is_weekend == 0) else 0 
                    input_data['Is_Weekend_Holiday'] = 1 if (is_holiday == 1 and is_weekend == 1) else 0 
                    input_data['Is_Long_Holiday'] = is_long_holiday
                    input_data['Is_Pre_Holiday'] = is_pre_holiday
                    input_data['Is_Post_Holiday'] = is_post_holiday
                    
                    if "Sunny" in future_weather: input_data['Weather_Clean_Sunny'] = 1
                    elif "Cloudy" in future_weather: input_data['Weather_Clean_Cloudy'] = 1
                    elif "Rainy" in future_weather: input_data['Weather_Clean_Rainy'] = 1
                        
                    selected_day = day_map[future_dayname]
                    day_col = f"DayName_Clean_{selected_day}"
                    if day_col in input_data.columns:
                        input_data[day_col] = 1
                    
                    zone_col = f"zone_{zone}"
                    if zone_col in input_data.columns:
                        input_data[zone_col] = 1
                        
                    pred_traffic = ai_model.predict(input_data)[0]
                    predictions[zone] = pred_traffic
                
                st.success("AI 분석 완료! 아래 예측 브리핑과 트래픽 곡선을 확인하세요.")
                
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
                except Exception as e:
                    pass

                st.markdown("""
                <div style="background-color: #F8FAFC; padding: 25px; border-radius: 15px; border-left: 5px solid #8B5CF6; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);">
                    <h3 style="color: #4C1D95; margin-top: 0;">📋 AI 매장 운영 브리핑</h3>
                """, unsafe_allow_html=True)
                
                holiday_txt = ""
                if is_holiday:
                    holiday_txt = f"(공휴일[{weekend_text}] / {'명절 연휴' if is_long_holiday else '일반'})"
                else:
                    holiday_txt = f"({weekend_text})"
                
                pre_post_txt = ""
                if is_pre_holiday: pre_post_txt = " 📌 **[공휴일 전날 효과 적용됨]**"
                if is_post_holiday: pre_post_txt = " 📌 **[공휴일 다음날 효과 적용됨]**"
                    
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
                
            except Exception as e:
                st.error(f"⚠️ AI 분석 중 오류가 발생했습니다. (사유: {e}) 새로 갱신된 'ai_forecaster.pkl' 파일이 깃허브에 잘 올라갔는지 확인해주세요!")

elif menu == "📍 센서(Sward) 위치":
    st.title("📍 매장 내 센서(Sward) 설치 위치")
    st.markdown("현재 마트에 설치된 센서 장비들의 위치와 구역 정보를 지도 위에서 확인합니다.")
    col1, col2 = st.columns([1, 3])
    with col1:
        with st.container(border=True):
            st.markdown("<h4 style='color: #047857; margin-top:0;'>💡 센서 연동 데이터</h4>", unsafe_allow_html=True)
            st.markdown("<p style='color: #475569; font-size: 14px; margin-bottom:0;'><code>swards (1).csv</code> 파일의 좌표를 기반으로 매장 지도 위에 실시간 매핑됩니다. 향후 센서가 추가/이동될 경우 CSV 파일만 교체하면 즉시 반영됩니다.</p>", unsafe_allow_html=True)
    with col2:
        try:
            sward_df = pd.read_csv('swards (1).csv')
            fig, ax = plt.subplots(figsize=(10, 7), dpi=200)
            img_path = 'map_image.jpg' 
            if os.path.exists(img_path):
                img = mpimg.imread(img_path)
                ax.imshow(img, extent=[0, 663, 500, 0], zorder=1)
            else:
                st.warning(f"지도 이미지('{img_path}')를 찾을 수 없습니다.")
                ax.set_xlim(0, 663); ax.set_ylim(500, 0); ax.invert_yaxis()
            ax.scatter(sward_df['x'], sward_df['y'], color='#EF4444', s=55, edgecolors='white', linewidth=2, zorder=2)
            for idx, row in sward_df.iterrows():
                ax.annotate(str(row['description']), (row['x'], row['y']), xytext=(5, 5), textcoords='offset points', fontsize=8, color='#1E3A8A', weight='bold', zorder=3, bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
            ax.axis('off')
            st.pyplot(fig)
        except FileNotFoundError: st.error("⚠️ 'swards (1).csv' 파일을 찾을 수 없습니다.")
        except Exception as e: st.error(f"오류가 발생했습니다: {e}")
