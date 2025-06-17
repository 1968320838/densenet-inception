from flask import Flask, request, jsonify, render_template_string
import base64
import io
from PIL import Image
import threading
import webbrowser
import time

app = Flask(__name__)

# 全局预测器
predictor = None

def init_predictor():
    """初始化预测器"""
    global predictor
    from mini.data.MNIST.model_inference import MNISTPredictor
    predictor = MNISTPredictor('best_model.pth')
    print("🌐 Web应用预测器初始化完成")

# HTML模板
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>MNIST数字识别</title>
    <style>
        body { font-family: Arial, sans-serif; text-align: center; padding: 20px; }
        #canvas { border: 2px solid #000; cursor: crosshair; }
        button { margin: 10px; padding: 10px 20px; font-size: 16px; }
        .result { font-size: 24px; margin: 20px; }
        .confidence { font-size: 18px; color: #666; }
    </style>
</head>
<body>
    <h1>🔢 MNIST手写数字识别</h1>
    <canvas id="canvas" width="280" height="280"></canvas><br>
    <button onclick="clearCanvas()">清除</button>
    <button onclick="predict()">识别</button>
    
    <div class="result" id="result">预测结果: -</div>
    <div class="confidence" id="confidence">置信度: -</div>
    
    <script>
        const canvas = document.getElementById('canvas');
        const ctx = canvas.getContext('2d');
        let isDrawing = false;
        
        // 设置画布
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = 'white';
        ctx.lineWidth = 15;
        ctx.lineCap = 'round';
        
        // 鼠标事件
        canvas.addEventListener('mousedown', startDrawing);
        canvas.addEventListener('mousemove', draw);
        canvas.addEventListener('mouseup', stopDrawing);
        canvas.addEventListener('mouseout', stopDrawing);
        
        // 触摸事件（移动设备支持）
        canvas.addEventListener('touchstart', handleTouch);
        canvas.addEventListener('touchmove', handleTouch);
        canvas.addEventListener('touchend', stopDrawing);
        
        function startDrawing(e) {
            isDrawing = true;
            draw(e);
        }
        
        function draw(e) {
            if (!isDrawing) return;
            
            const rect = canvas.getBoundingClientRect();
            const x = e.clientX - rect.left;
            const y = e.clientY - rect.top;
            
            ctx.lineTo(x, y);
            ctx.stroke();
            ctx.beginPath();
            ctx.moveTo(x, y);
        }
        
        function stopDrawing() {
            if (!isDrawing) return;
            isDrawing = false;
            ctx.beginPath();
        }
        
        function handleTouch(e) {
            e.preventDefault();
            const touch = e.touches[0];
            const mouseEvent = new MouseEvent(e.type === 'touchstart' ? 'mousedown' : 
                                            e.type === 'touchmove' ? 'mousemove' : 'mouseup', {
                clientX: touch.clientX,
                clientY: touch.clientY
            });
            canvas.dispatchEvent(mouseEvent);
        }
        
        function clearCanvas() {
            ctx.fillStyle = 'black';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            document.getElementById('result').textContent = '预测结果: -';
            document.getElementById('confidence').textContent = '置信度: -';
        }
        
        function predict() {
            const dataURL = canvas.toDataURL('image/png');
            const base64Data = dataURL.split(',')[1];
            
            fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ image: base64Data })
            })
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    document.getElementById('result').textContent = 
                        `预测结果: ${data.prediction}`;
                    document.getElementById('confidence').textContent = 
                        `置信度: ${(data.confidence * 100).toFixed(1)}%`;
                } else {
                    document.getElementById('result').textContent = '识别失败';
                }
            })
            .catch(error => {
                console.error('Error:', error);
                document.getElementById('result').textContent = '网络错误';
            });
        }
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """主页"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口"""
    try:
        data = request.get_json()
        image_data = data['image']
        
        # 解码base64图像
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        
        # 预测
        predicted_digit, confidence = predictor.predict_single(image)
        
        return jsonify({
            'success': True,
            'prediction': int(predicted_digit),
            'confidence': float(confidence)
        })
        
    except Exception as e:
        print(f"❌ 预测错误: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

def open_browser():
    """自动打开浏览器"""
    time.sleep(1)
    webbrowser.open('http://localhost:5000')

def run_web_app():
    """运行Web应用"""
    print("🚀 启动MNIST数字识别Web应用...")
    
    # 初始化预测器
    init_predictor()
    
    # 在新线程中打开浏览器
    threading.Thread(target=open_browser, daemon=True).start()
    
    print("🌐 Web应用已启动: http://localhost:5000")
    print("🖊️ 在浏览器中绘制数字进行识别")
    
    # 启动Flask应用
    app.run(host='0.0.0.0', port=5000, debug=False)

if __name__ == '__main__':
    run_web_app()