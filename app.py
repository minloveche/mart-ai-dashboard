import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.ndimage import gaussian_filter
import glob
import os
import platform

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

# --- [2. 데이터 로드 함수 (캐싱 적용으로 속도 최적화)] ---
@st.cache_data
def load_all_sessions():
    import os
    if os.path.exists("sessions_compressed.parquet"):
        return pd.read_parquet("sessions_compressed.parquet")
    return None

@st.cache_data
def load_trajectory():
    import os
    if os.path.exists("trajectory_super_light.parquet"):
        return pd.read_parquet("trajectory_super_light.parquet")
    return None

# ⭐ 이 아래 두 줄이 실수로 지워졌던 범인입니다! 반드시 있어야 합니다! ⭐
df_all = load_all_sessions()
df_traj = load_trajectory()

# --- [3. 사이드바 메뉴] ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3082/3082011.png", width=100)
st.sidebar.title("마트 AI 대시보드")
menu = st.sidebar.radio("메뉴를 선택하세요", ["📊 트래픽 요약", "🔥 정밀 히트맵", "🤖 AI 매대 시뮬레이터"])


# ====================================================================
# [메뉴 1] 트래픽 요약
# ====================================================================
if menu == "📊 트래픽 요약":
    st.title("📊 마트 트래픽 요약")

# ====================================================================
# [메뉴 1] 트래픽 요약
# ====================================================================
if menu == "📊 트래픽 요약":
    st.title("📊 마트 트래픽 요약")
    
    # 👇👇👇 [여기에 아래 코드를 복사해서 붙여넣으세요!] 👇👇👇
    
    st.markdown("""
    <div style="border: 3px dashed #CBD5E1; padding: 60px; text-align: center; border-radius: 15px; background-color: #F8FAFC; margin-bottom: 30px;">
        <h3 style="color: #334155; margin-bottom: 10px;">🔴 실시간 매장 트래픽 모니터링 (BETA)</h3>
        <p style="color: #64748B; font-size: 16px;">
            🚧 현재 개발 중인 기능입니다.<br>
            향후 CCTV 및 센서 데이터와 연동되어, 현재 매장 내 고객의 이동이 <b>실시간 점(Dot)</b>으로 표시될 공간입니다.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 👆👆👆 [여기까지 복사] 👆👆👆
    
    if df_all is not None and 'date' in df_all.columns:
        # ...(이하 기존 코드 동일)...
    
    if df_all is not None and 'date' in df_all.columns:
        import re
        
        # 1. 10월 1일, 10월 2일 순서대로 깔끔하게 정렬하기
        def sort_date(d):
            nums = re.findall(r'\d+', str(d))
            return int(nums[-1]) if nums else 0
            
        available_dates = sorted(df_all['date'].unique(), key=sort_date)
        
        # 2. 날짜 선택용 드롭다운 메뉴 만들기
        selected_date = st.selectbox("📅 조회할 날짜를 선택하세요:", ["전체 누적 보기"] + available_dates)
        
        # 3. 선택한 날짜에 맞춰서 데이터 필터링하기
        if selected_date == "전체 누적 보기":
            filtered_df = df_all
            st.markdown(f"### 📈 전체 누적 트래픽")
        else:
            filtered_df = df_all[df_all['date'] == selected_date]
            st.markdown(f"### 📈 {selected_date} 일자 트래픽")
            
        # 4. 필터링된 데이터로 화면에 보여주기
        if not filtered_df.empty:
            col1, col2, col3 = st.columns(3)
            total_users = filtered_df['real_user_id'].nunique()
            total_stays = filtered_df['stay_sec'].sum() / 3600
            top_zone = filtered_df['zone'].value_counts().index[0]

            col1.metric("해당 기간 방문 고객", f"{total_users:,} 명")
            col2.metric("고객 총 체류시간", f"{total_stays:,.0f} 시간")
            col3.metric("가장 붐빈 코너 1위", top_zone)

            st.markdown("### 🏆 구역별 방문 횟수 TOP 10")
            top10 = filtered_df['zone'].value_counts().head(10)
            st.bar_chart(top10)
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
    
    if df_traj is not None:
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

            df_exact = df_traj[(df_traj['x'] >= 0) & (df_traj['x'] <= 663) & (df_traj['y'] >= 0) & (df_traj['y'] <= 500)]
            heatmap_grid, _, _ = np.histogram2d(df_exact['y'], df_exact['x'], bins=[100, 132], range=[[0, 500], [0, 663]])
            heatmap_smoothed = gaussian_filter(heatmap_grid, sigma=blur_sigma)
            
            max_val = np.max(heatmap_smoothed)
            red_threshold = max_val * (red_sens / 100.0)
            vmin_level = max_val * 0.01

            im = ax.imshow(heatmap_smoothed, extent=[0, 663, 500, 0], cmap='Reds', alpha=0.6, 
                           zorder=3, interpolation='bilinear', vmin=vmin_level, vmax=red_threshold)
            ax.axis('off')
            st.pyplot(fig)
    else:
        st.error("'Real_Users_Trajectory1.csv' 파일이 필요합니다.")

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
                    # 1. 중심점 및 거리 계산 함수
                    def get_centers(z_dict):
                        return {z: np.array([(c['x_min']+c['x_max'])/2, (c['y_min']+c['y_max'])/2]) for z, c in z_dict.items()}
                    
                    def calc_dist(centers):
                        z_names = list(centers.keys())
                        dist_df = pd.DataFrame(np.zeros((len(z_names), len(z_names))), index=z_names, columns=z_names)
                        for z1 in z_names:
                            for z2 in z_names:
                                dist_df.loc[z1, z2] = np.linalg.norm(centers[z1] - centers[z2]) if z1 != z2 else 1.0
                        return dist_df

                    # 2. 현재 상태 학습
                    transition_counts = df_all.groupby(['zone', 'next_zone']).size().unstack(fill_value=0)
                    current_prob = transition_counts.div(transition_counts.sum(axis=1), axis=0).fillna(0)
                    current_traffic = df_all['zone'].value_counts()
                    
                    cur_centers = get_centers(ZONES)
                    cur_dist = calc_dist(cur_centers)
                    
                    # Intent(의도) = 확률 * 거리제곱
                    common_zones = current_prob.index.intersection(cur_dist.index)
                    intent_matrix = current_prob.loc[common_zones, common_zones] * (cur_dist.loc[common_zones, common_zones] ** 2)
                    
                    # 3. 매대 위치 Swap
                    new_zones = ZONES.copy()
                    new_zones[zone_A], new_zones[zone_B] = new_zones[zone_B], new_zones[zone_A]
                    
                    new_centers = get_centers(new_zones)
                    new_dist = calc_dist(new_centers)
                    
                    # 4. 새로운 확률 및 예측 계산
                    new_prob = intent_matrix / (new_dist.loc[common_zones, common_zones] ** 2)
                    new_prob = new_prob.div(new_prob.sum(axis=1), axis=0).fillna(0)
                    
                    pred_traffic = current_traffic.copy()
                    for _ in range(5): # 동선 5스텝 진행 예측
                        common_idx = pred_traffic.index.intersection(new_prob.index)
                        pred_traffic = pred_traffic[common_idx].dot(new_prob.loc[common_idx])
                    
                    # 5. 결과 시각화
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