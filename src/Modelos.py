import sklearn.datasets as skds
from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, KFold, cross_val_score, RandomizedSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score, root_mean_squared_error, roc_auc_score
from sklearn import metrics
from sklearn.pipeline import Pipeline
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from sklearn.decomposition import PCA

def xgb_mod(est:int,lr:float,boost:["gbtree","gblinear"],m_d:int,random:int):
    xgb_reg = xgboost.XGBRegressor(n_estimators = est, random_state=42)

    xgb_reg.fit(X_train, y_train)
    y_pred = xgb_reg.predict(X_test)

    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))
    print("MSE:", mean_squared_error(y_test, y_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred)))
    print("r2_score:", r2_score(y_test, y_pred))

    print("______________________________________________")