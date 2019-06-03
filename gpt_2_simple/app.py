import os
import gpt_2_simple as gpt2
import tensorflow as tf

from flask import Flask, request, make_response, jsonify

app = Flask(__name__)
run_name = 'poem-s-1'
graph = tf.Graph()

sess = gpt2.start_tf_sess(graph=graph)
gpt2.load_gpt2(sess, run_name=run_name)


@app.route('/', methods=['GET', 'POST'])
def root():
    if request.method == 'POST':
        body = request.form.get('b') or request.get_json().get('b')
        title = request.form.get('t') or request.get_json().get('t')
        if body or title:
            res = gpt2.generate(
                sess,
                run_name=run_name,
                length=1023,
                temperature=1,
                nsamples=1,
                truncate=True,
                top_k=40,
                body=body,
                title=title,
                return_as_list=True
            )
            return make_response(jsonify({'completion': res}))
        else:
            return make_response(jsonify({'error': 'start required'}), 422)
    return 'you too, oh palm!, are foreign to this soil'


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))
