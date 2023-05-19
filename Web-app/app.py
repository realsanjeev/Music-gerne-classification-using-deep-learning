import os
from flask import Flask, render_template, request
import music_recognizer

# create an instance of Flask class
app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def recognize():
    if request.method == 'POST':
        print(f"[INFO] POST method")
        try:
            print(f"[INFO] try method")
            music_file = request.files.get("file")
            print(f"[INFO] try method:  {music_file}")
        except Exception as err:
            return render_template('sorry.html', message=f'Please provide a valid file: {err}')
        
        if music_file is None:
            return render_template('sorry.html', message='Please provide a valid file.')
        
        save_path = os.path.join("static/music/", music_file.filename)
        music_file.save(save_path)
        
        print(f"Filename: {music_file.filename}")
        print(f"Saved as: {save_path}")
        
        # Perform music recognition here
        music_mfcc = music_recognizer.get_mfcc(save_path)
        if music_mfcc is None:
          return render_template('sorry.html', message='Error loading in server')
        predictions = music_recognizer.prediction(music_mfcc)
        path, result = music_recognizer.probability_graph_path(predictions)
        
        return render_template("index.html",
                                filename=path,
                                music_path=f"music/{music_file.filename}",
                                result=result)
    
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True)
