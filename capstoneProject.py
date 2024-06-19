import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import umap
from sklearn.metrics import silhouette_score
import altair as alt
import time

#Set theme
st.set_page_config(layout="centered",  initial_sidebar_state="expanded")


st.title('HVLV K-means Clustering Demo:point_up:') #title
st.subheader('K-means 기반 다품종 소량 모터 생산의 최적화 알고리즘') #subTitle 이모지 : https://streamlit-emoji-shortcodes-streamlit-app-gwckff.streamlit.app/
    
st.success('클러스터링할 데이터 셋의 파일 업로드를 먼저 하세요!')
# 파일 업로드 버튼
file = st.file_uploader("파일 선택", type=['xlsx'])
time.sleep(3)

# Excel 파일 업로드 확인
if file is not None:
    ext = file.name.split('.')[-1]
    data = pd.read_excel(file, engine='openpyxl', dtype=str) # 엑셀 로드
    tab1, tab2 = st.tabs(["품목구분과 사용설비와의 분류", "품목시리즈와 사용설비와의 분류"])

    def preProcessing(data, clustering_features, minY1, maxY1, minY2, maxY2):
        
        st.markdown("다품종 소량 모터 데이터")
        st.dataframe(data)
        st.markdown("데이터 전처리 후 데이터")
        data = data.dropna(subset=['ITEM_GUBUN', 'EQUIP_CD']) # 결측치 제거
        valid_item_gubun = ['001', '002', '003'] # 데이터의 노이즈 제거 (ITEM_GUBUN이 '001', '002', '003'이 아닌 경우 제거)
        data = data[data['ITEM_GUBUN'].isin(valid_item_gubun)]
    
        clustering_features = [feature for feature in clustering_features if feature in data.columns] # 선택한 변수가 실제로 존재하는지 확인
    
        for feature in clustering_features: # 범주형 변수 인코딩
            if data[feature].dtype == 'object':
                data[feature] = data[feature].astype('category').cat.codes

        st.dataframe(data, use_container_width=True)    

        scaler = StandardScaler()  # 데이터 스케일링
        scaled_features = scaler.fit_transform(data[clustering_features])
        df = pd.DataFrame(scaled_features, columns=clustering_features) # 클러스터링을 위한 데이터프레임 생성
        df.index = data.index  # 원래 데이터프레임의 인덱스를 유지

        sse = []
        silhouette_scores = []
        range_n_clusters = range(2, 8)

        for n_clusters in range_n_clusters:
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0)
            kmeans.fit(df)
            sse.append(kmeans.inertia_)
            silhouette_avg = silhouette_score(df, kmeans.labels_)
            silhouette_scores.append(silhouette_avg)

        
        st.markdown("아래 그래프는 최적의 클러스터 수를 찾기 위해 :green[엘보우 방법]을 시각화 한 것")
        sse_chart_data = pd.DataFrame({'n-cluster': range_n_clusters, 'see': sse})
        chart = alt.Chart(sse_chart_data).mark_line(color="#3dd56d").encode(  # Altair를 사용하여 차트 생성
            x=alt.X('n-cluster:O', title='Number of Clusters', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('see:Q', title='SEE', scale=alt.Scale(domain=[minY1, maxY1]))

        ).properties(
            width=704,
            height=350
        )
        st.altair_chart(chart)

        st.markdown("아래 그래프는 최적의 클러스터 수를 찾기 위해 :blue[실루엣 점수]를 시각화 한 것")
        silhouette_chart_data = pd.DataFrame({'n-cluster': range_n_clusters, 'silhouette_scores': silhouette_scores})
        chart = alt.Chart(silhouette_chart_data).mark_line(color="#60b4ff").encode(
            x=alt.X('n-cluster:O', title='Number of Clusters', axis=alt.Axis(labelAngle=0)),
            y=alt.Y('silhouette_scores:Q', title='Silhouette Scores', scale=alt.Scale(domain=[minY2, maxY2]))
        ).properties(
            width=704,
            height=350
        )
        st.altair_chart(chart)
        optimal_n_clusters = range_n_clusters[silhouette_scores.index(max(silhouette_scores))]  # 최적의 클러스터 수 결정
        
        return df, optimal_n_clusters, clustering_features
    
    def run_kmeans(df, n_clusters, clustering_features):
        kmeans = KMeans(n_clusters, init='k-means++', n_init=10, max_iter=300, random_state=0)
        df['cluster'] = kmeans.fit_predict(df)
        
        cluster_counts = df['cluster'].value_counts().sort_index() # 각 클러스터별 샘플 수 확인
        sizes = df['cluster'].map(cluster_counts) # 각 데이터 포인트의 크기 설정
        fig, ax = plt.subplots(figsize=(10, 8)) # 시각화

        # 스캐터 플롯
        sns.scatterplot(
            data=df, x=clustering_features[0], y=clustering_features[1], hue='cluster', 
            size=sizes, sizes=(50, 500),palette='Set1',alpha=0.6,ax=ax
        )

        # 타이틀 및 라벨 색상 설정
        plt.title('K-means++ Clustering on ITEM_SERIES, EQUIP_CD', color='white')
        plt.xlabel(clustering_features[0])
        plt.ylabel(clustering_features[1])

        # 레전드 설정
        #legend = plt.legend(title='Cluster')
        #plt.setp(legend.get_texts())  # 레전드 텍스트 색상 설정
        #plt.setp(legend.get_title())  # 레전드 타이틀 색상 설정
        plt.close(fig)  # Streamlit에서 사용하기 위해 figure를 닫습니다.

        # Streamlit에 그래프 표시
        st.pyplot(fig)


    with tab1:
        clustering_features = ['ITEM_SERIES', 'EQUIP_CD']
        second_clustering_features = ['EQUIP_CD', 'ITEM_SERIES', 'ITEM_GUBUN']
        df, optimal_n_clusters, clustering_features = preProcessing(data, clustering_features,100,1000,0.40,0.50)
        st.subheader('K-means Clustering Visualization')
        select_n_clusters = st.slider('Pick a Cluster Count', 0, optimal_n_clusters, value=optimal_n_clusters)
        st.write(run_kmeans(df,select_n_clusters,clustering_features))

    with tab2:
        clustering_features = ['ITEM_GUBUN', 'EQUIP_CD']
        second_clustering_features = ['ITEM_GUBUN', 'EQUIP_CD', 'ITEM_SERIES']
        df, optimal_n_clusters, clustering_features = preProcessing(data, clustering_features,0,350,0.700,0.950)
        st.subheader('K-means Clustering Visualization')
        select_n_clusters = st.slider('Pick a Cluster Count', 0, optimal_n_clusters, value=optimal_n_clusters)
        st.write(run_kmeans(df,select_n_clusters,clustering_features))
        
