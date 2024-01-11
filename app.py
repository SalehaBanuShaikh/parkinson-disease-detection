from flask import Flask,render_template,request
import numpy as np
import pickle
app = Flask(__name__)

svm_model=pickle.load(open('model.pkl','rb'))
@app.route('/')
def index():
   return render_template('index.html')

@app.route('/predict',methods=['post'])
def prediction():
    a = float(request.form['MDVP:Fo(Hz)'])
   #  b = float(request.form['MDVP:Fhi(Hz)'])
   #  c = float(request.form['MDVP:Flo(Hz)'])
   #  d = float(request.form['MDVP:Jitter(%)'])
   #  e = float(request.form['MDVP:Jitter(Abs)'])
   #  f = float(request.form['MDVP:RAP'])
   #  g = float(request.form['MDVP:PPQ'])
   #  h = float(request.form['Jitter:DDP'])
   #  i = float(request.form['MDVP:Shimmer'])
   #  j = float(request.form['MDVP:Shimmer(dB)'])
   #  k = float(request.form['Shimmer:APQ3'])
   #  l = float(request.form['Shimmer:APQ5'])
   #  m = float(request.form['MDVP:APQ'])
   #  n = float(request.form['Shimmer:DDA'])
   #  o = float(request.form['NHR'])
   #  p = float(request.form['HNR'])
   #  q = float(request.form['RPDE'])
   #  r = float(request.form['DFA'])
   #  s = float(request.form['spread1'])
   #  t = float(request.form['spread2'])
   #  u = float(request.form['D2'])
   #  v = float(request.form['PPE'])
    final = np.array([[a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a,a]])
   #  final = np.array([[a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v]])
    pred_model=svm_model.predict(final)

    if pred_model == 1:
       return render_template('output.html',output="HEALTH STATUS: POSITIVE")
    else:
       return render_template('output.html',output="HEALTH STATUS: NEGATIVE")









if __name__ =='__main__':
   app.run(debug=True, port=5500)
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
   
