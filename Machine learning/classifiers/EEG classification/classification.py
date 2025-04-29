import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler # data normalization
from sklearn.model_selection import train_test_split, GridSearchCV # find best parameters for model using K cross-validation
# import our classifiers
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# model evaluation
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# load data
def load_data(file:str) -> pd.DataFrame :
    """
    Load and show data information
    """
    data = pd.read_csv(file)
    print(data.head())
    print()
    print(data.info())
    print()
    print(data.describe())

    return data

# data preprocessing
def clean_data(data:pd.DataFrame):
    # data inspection
    print("Nan data:")
    print(data.isnull().sum())

    # replace Nan values with the mean value of the feature has Nan value
    if data.isnull().values.any():
        data = data.fillna(data.mean())

    # handle outliers
    z_scores = (data - data.mean()) / data.std()
    abs_z_scores = abs(z_scores) # returns the absolute values
    outliers = (abs_z_scores > 3).any(axis=1)

    # we can also consider if we need to enhance and merge features

    return data

def prepare_features_labels(data, target_column):
    # 分离特征和标签
    X = data.drop(columns=[target_column])
    y = data[target_column]
    
    # 检查每个类别的样本数量
    print("类别分布:")
    print(y.value_counts())
    
    return X, y

# normalization
def standardize_features(X_train:pd.DataFrame, X_test:pd.DataFrame):
    scaler = StandardScaler()
    # avoid data leeding but why?
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) 

    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_train.columns)

    return X_train_scaled, X_test_scaled, scaler

def select_features(X:pd.DataFrame, y:pd.DataFrame, threshold=0.3):
    correlations = []
    for col in X.columns:
        corr = abs(np.corrcoef(X[col],y)[0,1])
        correlations.append((col,corr))

    correlations.sort(key=lambda x: x[1], reverse=True)

    selected_features = [col for col, corr in correlations if corr >= threshold]

    return X[selected_features]

def train_svm(X_train, y_train):
    svm = SVC(probability=True)

    param_grid = {
        'C':[0.1,1,10,100],
        'gamma': ['scale','auto', 0.1, 0.01],
        'kernel': ['rbf','linear']
    }

    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    print("SVM 最佳参数:")
    print(grid_search.best_params_)
    print(f"SVM 最佳交叉验证分数: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name):
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"{model_name} 准确率: {accuracy:.4f}")
    
    # 分类报告
    print(f"{model_name} 分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{model_name} 混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.show()
    
    return accuracy, y_pred

def train_knn(X_train, y_train):
    # 创建KNN分类器
    knn = KNeighborsClassifier()
    
    # 定义参数网格
    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }
    
    # 使用网格搜索找到最佳参数
    grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', verbose=1)
    grid_search.fit(X_train, y_train)
    
    print("KNN 最佳参数:")
    print(grid_search.best_params_)
    print(f"KNN 最佳交叉验证分数: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_

# 使用方法
# knn_model = train_knn(X_train_scaled, y_train)

def full_pipeline(file_path, target_column, feature_threshold=0.3, test_size=0.2):
    # 加载数据
    data = load_data(file_path)
    
    # 清理数据
    cleaned_data = clean_data(data)
    
    # 准备特征和标签
    X, y = prepare_features_labels(cleaned_data, target_column)
    
    # 特征选择
    X_selected = select_features(X, y, threshold=feature_threshold)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

    # 标准化特征
    X_train_scaled, X_test_scaled, scaler = standardize_features(X_train, X_test)
    
    # 训练SVM模型
    print("\n训练SVM模型...")
    svm_model = train_svm(X_train_scaled, y_train)
    
    # 训练KNN模型
    print("\n训练KNN模型...")
    knn_model = train_knn(X_train_scaled, y_train)
    
    # 评估模型
    print("\n评估SVM模型:")
    svm_acc, _ = evaluate_model(svm_model, X_test_scaled, y_test, "SVM")
    
    print("\n评估KNN模型:")
    knn_acc, _ = evaluate_model(knn_model, X_test_scaled, y_test, "KNN")
    
    # 返回结果
    models = {"SVM": svm_model,"KNN": knn_model} # 
    accuracy = {"SVM": svm_acc, "KNN": knn_acc} # 
    
    return models, accuracy, scaler

# 使用方法
# models, accuracy, scaler = full_pipeline("your_eeg_data.csv", "target_column_name")

# 完整使用示例
if __name__ == "__main__":
    # 替换成您的文件路径和目标列名
    file_path = "EEGData.csv"  
    target_column = "target"  # 替换为您的目标列名
    
    # 运行完整流程
    models, accuracy, scaler = full_pipeline(file_path, target_column)
    
    # 打印最终结果
    print("\n最终结果:")
    for model_name, acc in accuracy.items():
        print(f"{model_name} 准确率: {acc:.4f}")
    
    # 找出最佳模型
    best_model = max(accuracy, key=accuracy.get)
    print(f"\n最佳模型是 {best_model} (准确率: {accuracy[best_model]:.4f})")