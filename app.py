from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        total_cases = float(request.form['total_cases'])
        Cases = pd.read_csv('2008_Cases_AI.csv')
        Deng = pd.read_csv('2008_Deng_AI.csv')
        Prueba_Data = pd.merge(Cases, Deng)
        
        X = Prueba_Data[['ndvi_se']]
        y = Prueba_Data[['total_cases']]
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict([[total_cases]])
        total_cases = y_pred[0][0]
        return render_template('index.html', total_cases=total_cases, plot='')
    else:
        Cases = pd.read_csv('2008_Cases_AI.csv')
        Deng = pd.read_csv('2008_Deng_AI.csv')
        Prueba_Data = pd.merge(Cases, Deng)
        X = Prueba_Data[['ndvi_se']]
        y = Prueba_Data[['total_cases']]
        reg = LinearRegression().fit(X, y)
        y_pred = reg.predict(X)

        plt.scatter(X, y, color='red')
        plt.plot(X, y_pred, color='blue')
        plt.xlabel('ndvi_se')
        plt.ylabel('total_cases')
        plt.title('Regresión lineal')

        buffer = io.BytesIO()
        plt.savefig(buffer, format='png')
        buffer.seek(0)
        plot_data = buffer.getvalue()
        plot = base64.b64encode(plot_data).decode()
        return render_template('index.html', plot=plot)

if __name__ == '__main__':
    app.run(debug=True)
