from numpy import asarray
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor
from matplotlib import pyplot

# 將時間序列資料集轉換為有監督的學習資料集  
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols = list()
	# 輸入序列 (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
	# 預測序列 (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
	# 資訊彙總
	agg = concat(cols, axis=1)
	# 刪除帶有空資料的行
	if dropnan:
		agg.dropna(inplace=True)
	return agg.values

# 將一個單變數資料集分割成訓練/測試集  
def train_test_split(data, n_test):
	return data[:-n_test, :], data[-n_test:, :]

# 擬合xgboost模型並進行一步預測  
def xgboost_forecast(train, testX):
	# 將列表轉換為陣列
	train = asarray(train)
	# 分成輸入和輸出列
	trainX, trainy = train[:, :-1], train[:, -1]
	# 擬合模型
	model = XGBRegressor(objective='reg:squarederror', n_estimators=1000)
	model.fit(trainX, trainy)
	# 做一步預測
	yhat = model.predict(asarray([testX]))
	return yhat[0]

# 單變數資料的前向驗證
def walk_forward_validation(data, n_test):
	predictions = list()
	# 切分資料集
	train, test = train_test_split(data, n_test)
	# 使用訓練資料集歷史資訊
	history = [x for x in train]
	# 步進測試集中的每個時間節點
	for i in range(len(test)):
		# 將測試行分割為輸入和輸出列
		testX, testy = test[i, :-1], test[i, -1]
		# 根據歷史擬合模型並進行預測
		yhat = xgboost_forecast(history, testX)
		# 在預測列表中儲存預測
		predictions.append(yhat)
		# 將實際觀察新增到下一個迴圈的歷史中  
		history.append(test[i])
		# 彙總結果
		print('>expected=%.1f, predicted=%.1f' % (testy, yhat))
	# 估計預測誤差
	error = mean_absolute_error(test[:, -1], predictions)
	return error, test[:, -1], predictions

# 載入資料集
series = read_csv('raw_1.csv', header=0, index_col=0)
values = series.values
# 將時間序列資料轉換為有監督學習  
data = series_to_supervised(values, n_in=10)
# 評估
mae, y, yhat = walk_forward_validation(data, 40000)
print('MAE: %.3f' % mae)
#計算R2
from sklearn.metrics import r2_score
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_absolute_error
import math
score = r2_score(y, yhat )
print("r2:", score)
RMSE = math.sqrt(mean_squared_error(y,yhat))
print("RMSE: " , RMSE)
MAE = mean_absolute_error(y,yhat )
print("MAE: " , MAE)
MAPE = mean_absolute_percentage_error(y, yhat)
print("MAPE: " , MAPE)

# 繪製預測結果圖
pyplot.figure(figsize=(16,8))
pyplot.plot(y, label='Expected')
pyplot.plot(yhat, marker = '*', label='Predicted')
pyplot.title("True vs Pred")
pyplot.legend(loc = 'best')
pyplot.show()