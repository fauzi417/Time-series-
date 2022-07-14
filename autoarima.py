
#AUTOMATION
import pmdarima as pm

model3 = pm.auto_arima(df3,
                      seasonal=True, m=7,
                      d=1, D=1,
                      start_p=1, start_q=1,
                      max_p=1, max_q=1,
                      max_P=1, max_Q=1,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True)

print(model3.summary())

import joblib
filename = 'model_arima.pkl'
joblib.dump(model,filename)
loaded_model = joblib.load(filename)

# Update the model
loaded_model.update(df_new)


