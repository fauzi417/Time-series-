
#AUTOMATION
import pmdarima as pm
from sklearn.model_selection import train_test_split

train,test=train_test_split(df,test_size=.2,shuffle=False)
model = pm.auto_arima(train,
                      test='adf',
                      d=1,
                      start_p=1, start_q=1,
                      max_p=1, max_q=1,
                      seasonal=True, m=7,
                      D=1,
                      start_P=0, start_Q=0
                      max_P=1, max_Q=1,
                      information_criterion='aic',
                      trace=True,
                      error_action='warn',
                      suppress_warnings=True,
                      stepwise = True,
                      n_job=-1)

print(model.summary())

import joblib
filename = 'model_arima.pkl'
joblib.dump(model,filename)
loaded_model = joblib.load(filename)

# Update the model
loaded_model.update(df_new)


