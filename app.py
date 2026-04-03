import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import glob
import os
import platform
import re
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

# --- [2. 데이터 로드 함수] ---
@st.cache_data
def load_all_sessions():
    if os.path.exists("sessions_compressed.parquet"):
        return pd.read_parquet("sessions_compressed.parquet")
    return None

@st.cache_data
def load_trajectory():
    if os.path.exists("trajectory_super_light.parquet"):
        return pd.read_parquet("trajectory_super_light.parquet")
    return None

# ⭐ 새롭게 추가된 날씨 데이터 로드 함수!
@st.cache_data
def load_weather():
    weather_dict = {}
    if os.path.exists("Day_Weather_Enhanced.csv"):
        try:
            df_w = pd.read_csv("Day_Weather_Enhanced.csv")
            for _, row in df_w.iterrows():
                date_str = str(row['Date']).strip()
                weather = str(row['Weather']).strip()
                holiday_flag = str(row['Holiday']).strip().lower()
                
                # 평일/휴일 및 날씨 이모지 변환
                holiday = "🔴 휴일" if holiday_flag == 'yes' else "🟢 평일"
                weather_lower = weather.lower()
                if "rain" in weather_lower: icon = "🌧️"
                elif "cloud" in weather_lower: icon = "☁️"
                elif "sun" in weather_lower or "clear" in weather_lower: icon = "☀️"
                else: icon = "🌤️"
                
                # "2025-10-01 [🌧️ Rainy | 🟢 평일]" 형태의 예쁜 이름표 완성
                weather_dict[date_str] = f"{date_str} [{icon} {weather} | {holiday}]"
        except Exception as e:
            pass
    return weather_dict

df_all = load_all_sessions()
df_traj = load_trajectory()
weather_info = load_weather() # 날씨 정보를 가져옵니다.

# ⭐ 드롭다운 메뉴의 글자만 바꿔주는 마법의 함수
def format_date_option(d):
    if d == "전체 누적 보기":
        return d
    return weather_info.get(str(d), str(d))

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
        <p style="color: #64748B; font-size: 16px;">
            🚧 현재 개발 중인 기능입니다.<br>
            향후 CCTV 및 센서 데이터와 연동되어, 현재 매장 내 고객의 이동이 <b>실시간 점(Dot)</b>으로 표시될 공간입니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if df_all is not None and 'date' in df_all.columns:
        def sort_date(d):
            nums = re.findall(r'\d+', str(d))
            return int(nums[-1]) if nums else 0
            
        available_dates = sorted(df_all['date'].unique(), key=sort_date)
        
        # ⭐ 여기에 format_func를 추가하여 화면에 날씨가 보이게 합니다.
        selected_date = st.selectbox("📅 조회할 날짜를 선택하세요:", ["전체 누적 보기"] + available_dates, format_func=format_date_option)
        
        if selected_date == "전체 누적 보기":
            filtered_df = df_all
            st.markdown(f"### 📈 전체 누적 트래픽")
            total_users = df_all.groupby('date')['real_user_id'].nunique().sum()
        else:
            filtered_df = df_all[df_all['date'] == selected_date]
            # ⭐ 제목에도 날씨와 휴일 정보가 뜨도록 합니다.
            display_title = weather_info.get(str(selected_date), str(selected_date))
            st.markdown(f"### 📈 {display_title} 트래픽")
            total_users = filtered_df['real_user_id'].nunique()
            
        if not filtered_df.empty:
            col1, col2, col3 = st.columns(3)
            
            total_stays = filtered_df['stay_sec'].sum() / 3600
            top_zone = filtered_df['zone'].value_counts().index[0]

            col1.metric("해당 기간 방문 고객 (연인원)", f"{total_users:,.0f} 명")
            col2.metric("고객 총 체류시간", f"{total_stays:,.0f} 시간")
            col3.metric("가장 붐빈 코너 1위", top_zone)

            st.markdown("### 🏆 구역별 전체 방문 횟수")
            all_zones = filtered_df['zone'].value_counts()
            
            df_zones = all_zones.reset_index()
            df_zones.columns = ['구역', '방문횟수']
            
            bars = alt.Chart(df_zones).mark_bar(cornerRadiusEnd=5).encode(
                x=alt.X('방문횟수:Q', title='방문 횟수 (회)', axis=alt.Axis(grid=False)),
                y=alt.Y('구역:N', sort='-x', title='', axis=alt.Axis(labelFontSize=13)),
                color=alt.Color('방문횟수:Q', scale=alt.Scale(scheme='blues'), legend=None),
                tooltip=['구역', '방문횟수']
            )
            
            text = bars.mark_text(
                align='left', baseline='middle', dx=5, fontSize=13, fontWeight='bold'
            ).encode(text=alt.Text('방문횟수:Q', format=','))
            
            final_chart = (bars + text).properties(height=alt.Step(35))
            st.altair_chart(final_chart, use_container_width=True)
            
        else:
            st.info("데이터가 없습니다.")
    else:
        st.error("데이터 파일에 날짜 정보가 없습니다. 데이터를 다시 압축해주세요.")

# ====================================================================
# [메뉴 2] 정밀 히트맵
# ====================================================================
elif menu == "🔥 정밀 히트맵":
    st.title("🔥 오리지널 구름 히트맵")
    st.markdown("슬라이더를 조절하여 히트맵의 붉은색 강도와 퍼짐 정도를 실시간으로 확인하세요.")
    
    if df_traj is not None and 'date' in df_traj.columns:
        def sort_date(d):
            nums = re.findall(r'\d+', str(d))
            return int(nums[-1]) if nums else 0
            
        available_dates = sorted(df_traj['date'].unique(), key=sort_date)
        
        # ⭐ 히트맵 메뉴의 드롭다운에도 날씨 포맷을 적용합니다.
        selected_date = st.selectbox("📅 조회할 날짜를 선택하세요:", ["전체 누적 보기"] + available_dates, key="heatmap_date", format_func=format_date_option)
        
        if selected_date == "전체 누적 보기":
            filtered_traj = df_traj
            st.markdown("### 📈 전체 누적 동선 히트맵")
        else:
            filtered_traj = df_traj[df_traj['date'] == selected_date]
            # ⭐ 제목에도 날씨 적용
            display_title = weather_info.get(str(selected_date), str(selected_date))
            st.markdown(f"### 📈 {display_title} 동선 히트맵")

        if not filtered_traj.empty:
            col1, col2 = st.columns([1, 3])
            with col1:
                st.markdown("#### 히트맵 컨트롤러")
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

                df_exact = filtered_traj[(filtered_traj['x'] >= 0) & (filtered_traj['x'] <= 663) & (filtered_traj['y'] >= 0) & (filtered_traj['y'] <= 500)]
                
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
                    st.warning("선택한 날짜에 유효한 동선(x, y) 데이터가 부족하여 히트맵을 그릴 수 없습니다.")
        else:
            st.info("선택한 날짜에 동선 데이터가 없습니다.")
    else:
        st.error("데이터에 날짜 정보가 없거나 'trajectory_super_light.parquet' 파일이 없습니다.")

# ====================================================================
# [메뉴 3] AI 매대 시뮬레이터 (마르코프 체인)
# ====================================================================
elif menu == "🤖 AI 매대 시뮬레이터":
    st.title("🤖 AI 매대 위치 최적화 시뮬레이터")
    st.markdown("두 매대의 위치를 바꾸면 고객 동선이 어떻게 변할지 AI가 예측합니다.")
    
    if df_all is not None:
        zones_list = list(ZONES.keys())
        
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
        st.info("💡 **센서(Sward) 연동 데이터**\n\n`swards (1).csv` 파일의 x, y 좌표를 기반으로 매장 지도 위에 실시간으로 센서 위치를 매핑합니다. 향후 센서가 추가/이동될 경우 CSV 파일만 교체하면 즉시 반영됩니다.")
    
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
            
            ax.scatter(sward_df['x'], sward_df['y'], color='red', s=45, edgecolors='white', linewidth=1.5, zorder=2)
            
            for idx, row in sward_df.iterrows():
                ax.annotate(str(row['description']), 
                            (row['x'], row['y']), 
                            xytext=(5, 5), textcoords='offset points', 
                            fontsize=8, color='#1E3A8A', weight='bold', zorder=3)
                
            ax.axis('off')
            st.pyplot(fig)
            
        except FileNotFoundError:
            st.error("⚠️ 'swards (1).csv' 파일을 찾을 수 없습니다.")
            st.info("이 기능이 작동하려면 깃허브(GitHub) 창고에 'swards (1).csv' 파일이 업로드되어 있어야 합니다!")
        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")