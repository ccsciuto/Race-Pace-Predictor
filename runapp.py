from flask import Flask, render_template,request
import pickle
import numpy as np
app = Flask(__name__)

model=pickle.load(open('run_model.pickle','rb'))
@app.route('/')
def hello_world():
    return render_template('web.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    prediction=model.predict_proa(final)
    output='{0:.{1}f}'.format(prediction[0][1],2)


    if output == output:
        return render_template('web.html',pred='Predicted Pace is {}'.format(output))

if __name__ == '__main__':
    app.run()