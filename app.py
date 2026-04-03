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

# --- [1. 기본 설정 및 한글 폰트] ---
st.set_page_config(page_title="Retail AI Dashboard", page_icon="🛒", layout="wide")

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# 구역(Zones) 좌표 데이터
ZONES = {
    '행사(1)': {'x_min': 489, 'x_max': 528, 'y_min': 301, 'y_max': 374},
    '문구(1)': {'x_min': 528, 'x_max': 587, 'y_min': 303, 'y_max': 372},
    '장난감': {'x_min': 494, 'x_max': 560, 'y_min': 398, 'y_max': 485},
    '침구': {'x_min': 420, 'x_max': 494, 'y_min': 396, 'y_max': 482},
    '보수용품': {'x_min': 239, 'x_max': 421, 'y_min': 397, 'y_max': 493},
    '음료': {'x_min': 183, 'x_max': 239, 'y_min': 397, 'y_max': 452},
    '주류': {'x_min': 99,  'x_max': 183, 'y_min': 389, 'y_max': 452},
    '식품코너': {'x_min': 42,  'x_max': 102, 'y_min': 313, 'y_max': 406},
    '과자': {'x_min': 127, 'x_max': 186, 'y_min': 301, 'y_max': 374},
    '화장품': {'x_min': 487, 'x_max': 586, 'y_min': 163, 'y_max': 267},
    '반찬/소스': {'x_min': 302, 'x_max': 372, 'y_min': 161, 'y_max': 223},
    '커피/차': {'x_min': 208, 'x_max': 285, 'y_min': 266, 'y_max': 300},
    '주방용품': {'x_min': 298, 'x_max': 393, 'y_min': 300, 'y_max': 373},
    '자동차용품': {'x_min': 389, 'x_max': 415, 'y_min': 298, 'y_max': 376},
    '문구(2)': {'x_min': 420, 'x_max': 468, 'y_min': 298, 'y_max': 374},
    '냉동식품': {'x_min': 128, 'x_max': 189, 'y_min': 163, 'y_max': 299},
    '퍼스널케어': {'x_min': 371, 'x_max': 469, 'y_min': 161, 'y_max': 230},
    '축산': {'x_min': 59,  'x_max': 105, 'y_min': 169, 'y_max': 297},
    '수산': {'x_min': 61,  'x_max': 159, 'y_min': 73,  'y_max': 138},
    '속옷': {'x_min': 463, 'x_max': 536, 'y_min': 56,  'y_max': 135},
    '스포츠': {'x_min': 603, 'x_max': 633, 'y_min': 57,  'y_max': 137},
    '스포츠(2)': {'x_min': 537, 'x_max': 602, 'y_min': 57,  'y_max': 137},
    '제임스딘': {'x_min': 429, 'x_max': 451, 'y_min': 73,  'y_max': 137},
    '곡물/건조식품': {'x_min': 293, 'x_max': 426, 'y_min': 71,  'y_max': 137},
    '채소/계란/과일': {'x_min': 158, 'x_max': 292, 'y_min': 81,  'y_max': 138},
    '라면': {'x_min': 209, 'x_max': 305, 'y_min': 161, 'y_max': 227},
    '행사(2)': {'x_min': 207, 'x_max': 284, 'y_min': 223, 'y_max': 265},
    '시리얼': {'x_min': 286, 'x_max': 307, 'y_min': 229, 'y_max': 295},
    '휴지': {'x_min': 207, 'x_max': 294, 'y_min': 302, 'y_max': 375},
    '홈데코': {'x_min': 236, 'x_max': 322, 'y_min': 399, 'y_max': 493}
}

# --- [2. 데이터 로드 및 지능형 날짜 매칭] ---
@st.cache_data
def load_all_sessions():
    files = glob.glob("Zone_Visit_Sessions*.*") + glob.glob("sessions_compressed.*")
    files = [f for f in files if f.endswith('.parquet') or f.endswith('.csv')]
    if not files: return None
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f) if f.endswith('.parquet') else pd.read_csv(f)
            # 파일명(2025_10_1)에서 날짜 추출
            match = re.search(r'(\d{4})_(\d{1,2})_(\d{1,2})', f)
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
            match = re.search(r'(\d{4})_(\d{1,2})_(\d{1,2})', f)
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
                day_num = index + 1 
                date_str = str(row['Date']).strip()
                weather = str(row['Weather']).strip()
                holiday_flag = str(row['Holiday']).strip().lower()
                holiday = "🔴 휴일" if holiday_flag == 'yes' else "🟢 평일"
                weather_lower = weather.lower()
                if "rain" in weather_lower: icon = "🌧️"
                elif "cloud" in weather_lower: icon = "☁️"
                elif "sun" in weather_lower or "clear" in weather_lower: icon = "☀️"
                else: icon = "🌤️"
                weather_dict[day_num] = f"{date_str} [{icon} {weather} | {holiday}]"
        except: pass
    return weather_dict

df_all = load_all_sessions()
df_traj = load_trajectory()
weather_info = load_weather()

# ⭐ [핵심] 어떤 형식이든 숫자만 뽑아서 날짜를 매칭하는 무적의 함수
def safe_date_match(val, target):
    def get_day_num(x):
        nums = re.findall(r'\d+', str(x).split('.')[0])
        return int(nums[-1]) if nums else None
    
    v1 = get_day_num(val)
    v2 = get_day_num(target)
    if v1 is not None and v2 is not None:
        return v1 == v2
    return str(val).strip() == str(target).strip()

def format_date_option(d):
    if d == "전체 누적 보기": return d
    try:
        day_num = int(str(d).split('-')[-1])
        return weather_info.get(day_num, str(d))
    except: return str(d)

# --- [3. 사이드바 메뉴] ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3082/3082011.png", width=100)
st.sidebar.title("마트 AI 대시보드")
menu = st.sidebar.radio("메뉴를 선택하세요", ["📊 트래픽 요약", "🔥 정밀 히트맵", "🤖 AI 매대 시뮬레이터", "📍 센서(Sward) 위치"])

# ====================================================================
# [메뉴 1] 트래픽 요약
# ====================================================================
if menu == "📊 트래픽 요약":
    st.title("📊 마트 트래픽 요약")
    
    st.markdown("""
    <div style="border: 3px dashed #CBD5E1; padding: 60px; text-align: center; border-radius: 15px; background-color: #F8FAFC; margin-bottom: 30px;">
        <h3 style="color: #334155; margin-bottom: 10px;">🔴 실시간 매장 트래픽 모니터링 (BETA)</h3>
        <p style="color: #64748B; font-size: 16px;">🚧 현재 개발 중인 기능입니다.</p>
    </div>
    """, unsafe_allow_html=True)

    if df_all is not None and 'date' in df_all.columns:
        available_dates = sorted(df_all['date'].unique().tolist())
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

            st.markdown("---")
            
            # ⭐ [해결] 누적과 개별 날짜 모두 예쁘게 나오는 그래프 구역
            st.markdown("### 🌊 시간대별 매장 정밀 트래픽 흐름 (10분 단위)")
            try:
                trend_df = pd.read_csv("time_trend_light.csv")
                
                if selected_date == "전체 누적 보기":
                    # 누적일 때는 시간대별로 모두 합침
                    plot_data = trend_df.groupby('time_str')['visitors'].sum().reset_index()
                    y_title = '총 누적 방문객 수 (명)'
                else:
                    # 특정 날짜일 때는 무적의 매칭 함수 사용
                    plot_data = trend_df[trend_df['date'].apply(lambda x: safe_date_match(x, selected_date))]
                    y_title = '동시 체류 방문객 수 (명)'

                if not plot_data.empty:
                    base_date = pd.to_datetime("2026-01-01")
                    plot_data['시간'] = pd.to_datetime(base_date.strftime('%Y-%m-%d') + ' ' + plot_data['time_str'])
                    
                    chart = alt.Chart(plot_data).mark_area(
                        interpolate='monotone', color='#60A5FA', opacity=0.3
                    ).encode(
                        x=alt.X('시간:T', title='시간', axis=alt.Axis(format='%H:%M')),
                        y=alt.Y('visitors:Q', title=y_title),
                        tooltip=['time_str', 'visitors']
                    ) + alt.Chart(plot_data).mark_line(
                        interpolate='monotone', color='#2563EB', strokeWidth=3
                    ).encode(
                        x=alt.X('시간:T'),
                        y=alt.Y('visitors:Q')
                    )
                    st.altair_chart(chart.properties(height=350).interactive(), use_container_width=True)
                else:
                    st.info("💡 선택하신 날짜의 시간대별 트래픽 데이터가 없습니다.")
            except Exception as e:
                st.error(f"그래프 생성 중 오류: {e}")

            st.markdown("---")
            # 구역별 방문 횟수
            st.markdown("### 🏆 구역별 전체 방문 횟수")
            df_zones = filtered_df['zone'].value_counts().reset_index()
            df_zones.columns = ['구역', '방문횟수']
            bars = alt.Chart(df_zones).mark_bar(cornerRadiusEnd=5).encode(
                x=alt.X('방문횟수:Q', title='방문 횟수 (회)'),
                y=alt.Y('구역:N', sort='-x', title=''),
                color=alt.Color('방문횟수:Q', scale=alt.Scale(scheme='blues'), legend=None)
            )
            st.altair_chart(bars.properties(height=alt.Step(35)), use_container_width=True)