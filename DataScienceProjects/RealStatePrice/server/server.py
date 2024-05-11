from flask import Flask, request, jsonify
import util
app = Flask(__name__)


@app.route('/hello')
def hello():
    return '<h1>Hi</h1>'


@app.route('/dsfd', methods=['GET', 'POST'])
def classify_mage():
    return '<h1>Hi</h1>'


@app.route('/get_location_names')
def get_location_names():
    response = jsonify({'location': util.get_location_names()})
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


@app.route('/get_est_price', methods=['POST'])
def get_est_price():
    # 'ramamurthy nagar', 2100, 2, 3
    r1 = request.form
    sqft = float(r1['sqft'])
    location = r1['location']
    bed = int(r1['bed'])
    bath = int(r1['bath'])
    response = jsonify({'est_price': util.get_estimated_price(
        location=location, sqft=sqft, bath=bath, bed=bed)})
    response.headers.add('Access-Control-Allow-Origin', '*')

    return response


if __name__ == '__main__':
    util.load_saved_artiefacts()
    app.run(port=5000)
