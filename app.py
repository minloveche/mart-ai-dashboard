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

# --- [1. 기본 설정 및 한글 폰트] ---
st.set_page_config(page_title="Retail AI Dashboard", page_icon="🛒", layout="wide")

if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
elif platform.system() == 'Darwin':
    plt.rc('font', family='AppleGothic')
else:
    plt.rc('font', family='NanumGothic')
plt.rcParams['axes.unicode_minus'] = False

# ⭐ [디자인 업그레이드!] 
custom_css = """
<style>
    .stApp { background-color: #F8FAFC; }
    [data-testid="stSidebar"] { background-color: #1E293B !important; }
    [data-testid="stSidebar"] * { color: #F1F5F9 !important; }
    h1, h2, h3 { color: #0F172A; font-weight: 800 !important; letter-spacing: -0.5px; }
    [data-testid="stMetric"] {
        background-color: #FFFFFF; padding: 20px; border-radius: 15px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05); border: 1px solid #E2E8F0; text-align: center; transition: transform 0.2s;
    }
    [data-testid="stMetric"]:hover { transform: translateY(-5px); }
    [data-testid="stMetricLabel"] { font-size: 15px; color: #64748B; font-weight: 600; }
    [data-testid="stMetricValue"] { font-size: 36px; color: #2563EB; font-weight: 900; }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #FFFFFF !important; border-radius: 15px !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05) !important; border: 1px solid #E2E8F0 !important; padding: 15px !important;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

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

def safe_date_match(val, target):
    if '-' in str(val) and '-' in str(target):
        return str(val).strip() == str(target).strip()
    def get_day_num(x):
        nums = re.findall(r'\d+', str(x).split('.')[0])
        return int(nums[-1]) if nums else None
    
    v1 = get_day_num(val)
    v2 = get_day_num(target)
    if v1 is not None and v2 is not None:
        return v1 == v2
    return str(val).strip() == str(target).strip()

def sort_date_smart(d):
    nums = re.findall(r'\d+', str(d))
    if not nums: return 99999999
    if len(nums) >= 3:
        return int(f"{nums[0]}{int(nums[1]):02d}{int(nums[2]):02d}")
    return int(nums[-1])

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
    <div style="background: linear-gradient(135deg, #EFF6FF 0%, #DBEAFE 100%); padding: 30px; border-radius: 15px; border-left: 5px solid #3B82F6; margin-bottom: 30px; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
        <h4 style="color: #1E3A8A; margin-top: 0;">🔴 실시간 매장 트래픽 모니터링 (BETA)</h4>
        <p style="color: #475569; font-size: 15px; margin-bottom: 0;">
            🚧 현재 개발 중인 기능입니다.<br>
            향후 CCTV 및 센서 데이터와 연동되어, 현재 매장 내 고객의 이동이 <b>실시간 점(Dot)</b>으로 표시될 공간입니다.
        </p>
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
                    
                    chart = alt.Chart(plot_data).mark_area(
                        interpolate='monotone', color='#93C5FD', opacity=0.4
                    ).encode(
                        x=alt.X('시간:T', title='시간', axis=alt.Axis(format='%H:%M', labelColor='#475569')),
                        y=alt.Y('visitors:Q', title=y_title, axis=alt.Axis(labelColor='#475569')),
                        tooltip=[alt.Tooltip('시간:T', format='%H:%M', title='시간대'), alt.Tooltip('visitors:Q', title='방문객 수')]
                    ) + alt.Chart(plot_data).mark_line(
                        interpolate='monotone', color='#3B82F6', strokeWidth=3
                    ).encode(
                        x=alt.X('시간:T'),
                        y=alt.Y('visitors:Q')
                    )
                    st.altair_chart(chart.properties(height=380).interactive(), use_container_width=True)
                else:
                    st.info("💡 선택하신 날짜의 시간대별 트래픽 데이터가 없습니다.")
            except Exception as e:
                st.error(f"그래프 생성 중 오류: {e}")

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
            text = bars.mark_text(
                align='left', baseline='middle', dx=5, fontSize=13, fontWeight='bold', color='#1E293B'
            ).encode(text=alt.Text('방문횟수:Q', format=','))
            
            final_chart = (bars + text).properties(height=alt.Step(35))
            st.altair_chart(final_chart, use_container_width=True)
            
        else:
            st.info("데이터가 없습니다.")
    else:
        st.error("데이터 파일에 날짜 정보가 없습니다. 깃허브에 데이터 파일이 존재하는지 확인해주세요.")

# ====================================================================
# [메뉴 2] 정밀 히트맵
# ====================================================================
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
                    
                    # ⭐ [새 기능!] 딱 한 개만 고르는 스냅샷 슬라이더 (10분 단위로 움직임)
                    selected_time = st.slider(
                        "⏰ 특정 시간 스냅샷 보기",
                        min_value=datetime.time(9, 0),  # 오픈
                        max_value=datetime.time(22, 50), # 마감 직전 10분
                        value=datetime.time(15, 0), # 기본값 오후 3시
                        step=datetime.timedelta(minutes=10), # 10분 단위로만 딱딱 맞춰서 이동!
                        format="HH:mm"
                    )
                    
                    # 현재 보고 있는 시간을 보기 좋게 표시해 줍니다.
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

                # ⭐ 스냅샷 시간에 맞춰 동선 데이터 필터링!
                df_exact = filtered_traj[(filtered_traj['x'] >= 0) & (filtered_traj['x'] <= 663) & (filtered_traj['y'] >= 0) & (filtered_traj['y'] <= 500)].copy()
                
                if 'time_index' in df_exact.columns and not df_exact.empty:
                    time_idx = pd.to_numeric(df_exact['time_index'], errors='coerce').fillna(0)
                    total_secs = (time_idx * 10) % 86400
                    
                    # 선택한 시간을 '초'로 변환 (오후 3시면 15*3600 = 54000초)
                    target_sec = selected_time.hour * 3600 + selected_time.minute * 60
                    
                    # 정확히 선택한 시간부터 10분(600초) 동안에 해당하는 데이터만 남기기!
                    df_exact = df_exact[(total_secs >= target_sec) & (total_secs < target_sec + 600)]

                if len(df_exact) > 0:
                    heatmap_grid, _, _ = np.histogram2d(df_exact['y'], df_exact['x'], bins=[100, 132], range=[[0, 500], [0, 663]])
                    heatmap_smoothed = gaussian_filter(heatmap_grid, sigma=blur_sigma)
                    
                    max_val = np.max(heatmap_smoothed)
                    if max_val > 0:
                        red_threshold = max_val * (red_sens / 100.0)
                        vmin_level = max_val * 0.01

                        im = ax.imshow(heatmap_smoothed, extent=[0, 663, 500, 0], cmap='Reds', alpha=0.6, 
                                       zorder=3, interpolation='bilinear', vmin=vmin_level, vmax=red_threshold)
                    ax.axis('off')
                    st.pyplot(fig)
                else:
                    st.warning("⚠️ 선택하신 스냅샷 시간대에는 고객 동선 데이터가 없습니다.")
        else:
            st.info("선택한 날짜에 동선 데이터가 없습니다.")
    else:
        st.error("데이터에 날짜 정보가 없거나 궤적(Trajectory) 파일이 없습니다.")

# ====================================================================
# [메뉴 3] AI 매대 시뮬레이터 (마르코프 체인)
# ====================================================================
elif menu == "🤖 AI 매대 시뮬레이터":
    st.title("🤖 AI 매대 위치 최적화 시뮬레이터")
    st.markdown("두 매대의 위치를 바꾸면 고객 동선이 어떻게 변할지 AI가 예측합니다.")
    
    if df_all is not None:
        zones_list = list(ZONES.keys())
        
        with st.container(border=True):
            st.markdown("<h4 style='color: #1E293B; margin-top:0;'>🔄 매대 위치 스왑 설정</h4>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)
            with col1:
                zone_A = st.selectbox("위치를 바꿀 매대 1", zones_list, index=zones_list.index('과자'))
            with col2:
                zone_B = st.selectbox("위치를 바꿀 매대 2", zones_list, index=zones_list.index('라면'))
                
            if st.button("🚀 시뮬레이션 시작", use_container_width=True):
                if zone_A == zone_B:
                    st.warning("서로 다른 두 매대를 선택해 주세요.")
                else:
                    with st.spinner("AI가 고객 이동 의도(Intent)를 계산 중입니다..."):
                        def get_centers(z_dict):
                            return {z: np.array([(c['x_min']+c['x_max'])/2, (c['y_min']+c['y_max'])/2]) for z, c in z_dict.items()}
                        
                        def calc_dist(centers):
                            z_names = list(centers.keys())
                            dist_df = pd.DataFrame(np.zeros((len(z_names), len(z_names))), index=z_names, columns=z_names)
                            for z1 in z_names:
                                for z2 in z_names:
                                    dist_df.loc[z1, z2] = np.linalg.norm(centers[z1] - centers[z2]) if z1 != z2 else 1.0
                            return dist_df

                        transition_counts = df_all.groupby(['zone', 'next_zone']).size().unstack(fill_value=0)
                        current_prob = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
                        current_traffic = df_all['zone'].value_counts()
                        
                        cur_centers = get_centers(ZONES)
                        cur_dist = calc_dist(cur_centers)
                        
                        common_zones = current_prob.index.intersection(cur_dist.index)
                        intent_matrix = current_prob.loc[common_zones, common_zones] * (cur_dist.loc[common_zones, common_zones] ** 2)
                        
                        new_zones = ZONES.copy()
                        new_zones[zone_A], new_zones[zone_B] = new_zones[zone_B], new_zones[zone_A]
                        
                        new_centers = get_centers(new_zones)
                        new_dist = calc_dist(new_centers)
                        
                        new_prob = intent_matrix / (new_dist.loc[common_zones, common_zones] ** 2)
                        new_prob = new_prob.div(new_prob.sum(axis=1), axis=0).fillna(0)
                        
                        pred_traffic = current_traffic.copy()
                        for _ in range(5):
                            common_idx = pred_traffic.index.intersection(new_prob.index)
                            pred_traffic = pred_traffic[common_idx].dot(new_prob.loc[common_idx])
                        
                        st.success("예측 완료!")
                        res_col1, res_col2 = st.columns(2)
                        
                        old_a = current_traffic.get(zone_A, 0)
                        new_a = pred_traffic.get(zone_A, 0)
                        delta_a = new_a - old_a
                        res_col1.metric(f"[{zone_A}] 코너 예측 방문객", f"{new_a:,.0f}명", f"{delta_a:,.0f}명 ({(delta_a/old_a)*100:.1f}%)")
                        
                        old_b = current_traffic.get(zone_B, 0)
                        new_b = pred_traffic.get(zone_B, 0)
                        delta_b = new_b - old_b
                        res_col2.metric(f"[{zone_B}] 코너 예측 방문객", f"{new_b:,.0f}명", f"{delta_b:,.0f}명 ({(delta_b/old_b)*100:.1f}%)")
    else:
         st.error("데이터 파일이 필요합니다.")

# ====================================================================
# [메뉴 4] 센서(Sward) 위치 확인
# ====================================================================
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
                ax.annotate(str(row['description']), 
                            (row['x'], row['y']), 
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=8, color='#1E3A8A', weight='bold', zorder=3,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.7))
                
            ax.axis('off')
            st.pyplot(fig)
            
        except FileNotFoundError:
            st.error("⚠️ 'swards (1).csv' 파일을 찾을 수 없습니다.")
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")