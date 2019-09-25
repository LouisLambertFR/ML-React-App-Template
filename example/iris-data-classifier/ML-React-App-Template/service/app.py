from flask import Flask, request, jsonify, make_response
from flask_restplus import Api, Resource, fields
from sklearn.externals import joblib
import numpy as np
import sys

flask_app = Flask(__name__)
app = Api(app = flask_app, 
		  version = "1.0", 
		  title = "Automated Home Valuation", 
		  description = "Predict the price of Homes")

name_space = app.namespace('prediction', description='Prediction APIs')

model = app.model('Prediction params', 
				  {'Surfacereellebati': fields.Float(required = True, 
							       				description="Surface reelle bati",
    					  				 	   help="Surface reelle bati cannot be blank"),
				  'Nombrepiecesprincipales': fields.Float(required = True, 
				  							description="Nombre pieces principales", 
    					  				 	   help="Nombre pieces principales cannot be blank")})

regressor = joblib.load('regressor.joblib')

@name_space.route("/")
class MainClass(Resource):

	def options(self):
		response = make_response()
		response.headers.add("Access-Control-Allow-Origin", "*")
		response.headers.add('Access-Control-Allow-Headers', "*")
		response.headers.add('Access-Control-Allow-Methods', "*")
		return response

	@app.expect(model)		
	def post(self):
		try: 
			formData = request.json
			data = [val for val in formData.values()]
			prediction = regressor.predict(np.array(data).reshape(1, -1))
			types = { 0: "Iris Setosa", 1: "Iris Versicolour ", 2: "Iris Virginica"}
			response = jsonify({
				"statusCode": 200,
				"status": "Prediction made",
				"result": "The price of your home is: " + types[prediction[0]]
				})
			response.headers.add('Access-Control-Allow-Origin', '*')
			return response
		except Exception as error:
			return jsonify({
				"statusCode": 500,
				"status": "Could not make prediction",
				"error": str(error)
			})
