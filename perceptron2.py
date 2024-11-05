import math
import random

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01, seed=42):
        random.seed(seed)
        
        # Inicialización de pesos y biases para la capa oculta
        self.weights_input_to_hidden = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.bias_hidden = [random.uniform(-1, 1) for _ in range(hidden_size)]
        
        # Inicialización de pesos y biases para la capa de salida
        self.weights_hidden_to_output = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
        self.bias_output = [random.uniform(-1, 1) for _ in range(output_size)]
        
        # Tasa de aprendizaje
        self.learning_rate = learning_rate

    def sigmoid(self, x):
        """Función sigmoide aplicada a las neuronas de la capa oculta."""
        return 1 / (1 + math.exp(-x))
    
    def sigmoid_derivative(self, x):
        """Derivada de la función sigmoide, utilizada para el cálculo de gradientes."""
        return x * (1 - x)
    
    def softmax(self, z):
        """Función softmax aplicada en la capa de salida."""
        exps = [math.exp(i) for i in z]
        sum_exps = sum(exps)
        return [exp_i / sum_exps for exp_i in exps]

    

    def capa_oculta(self, inputs):
        """Cálculo de la salida de la capa oculta."""
        hidden_layer_output = []
        for i in range(len(self.weights_input_to_hidden)):
            total = sum(w * inp for w, inp in zip(self.weights_input_to_hidden[i], inputs)) + self.bias_hidden[i]
            hidden_layer_output.append(self.sigmoid(total))
        return hidden_layer_output

    def capa_salida(self, hidden_layer_output):
        """Cálculo de la salida de la capa de salida."""
        output_layer_input = []
        for i in range(len(self.weights_hidden_to_output)):
            total = sum(w * hidden_out for w, hidden_out in zip(self.weights_hidden_to_output[i], hidden_layer_output)) + self.bias_output[i]
            output_layer_input.append(self.sigmoid(total))  # Aplicar sigmoide en la salida
        return output_layer_input

    def calculate_mse(self, outputs, expected_outputs):
        """Cálculo del error cuadrático medio."""
        return 0.5 * sum((expected - output) ** 2 for expected, output in zip(expected_outputs, outputs))
    
    def predict(self, inputs):
        """Realiza una predicción para una entrada dada."""
        hidden_output = self.capa_oculta(inputs)
        output_layer_input = self.capa_salida(hidden_output)
        output = self.softmax(output_layer_input)
        return output.index(max(output))

    def train(self, inputs, expected_outputs):
        """Entrenamiento utilizando gradientes y el error cuadrático medio."""
        
        # Forward Pass
        hidden_output = self.capa_oculta(inputs)
        actual_outputs = self.capa_salida(hidden_output)
        
        # Cálculo del error cuadrático medio (ECM)
        error = self.calculate_mse(actual_outputs, expected_outputs)
        
        # Cálculo del error en la capa de salida
        output_errors = [(expected - actual) * self.sigmoid_derivative(actual) 
                         for expected, actual in zip(expected_outputs, actual_outputs)]
        
        # Ajuste de pesos y biases para la capa de salida
        for i in range(len(self.weights_hidden_to_output)):
            for j in range(len(self.weights_hidden_to_output[i])):
                # Ajuste de pesos con gradientes
                self.weights_hidden_to_output[i][j] += self.learning_rate * output_errors[i] * hidden_output[j]
            # Ajuste de bias de la capa de salida
            self.bias_output[i] += self.learning_rate * output_errors[i]

        # Retropropagación del error a la capa oculta
        hidden_errors = []
        for i in range(len(self.weights_input_to_hidden)):
            error = sum(self.weights_hidden_to_output[j][i] * output_errors[j] 
                        for j in range(len(self.weights_hidden_to_output)))
            hidden_errors.append(error * self.sigmoid_derivative(hidden_output[i]))

        # Ajuste de pesos y biases para la capa oculta
        for i in range(len(self.weights_input_to_hidden)):
            for j in range(len(self.weights_input_to_hidden[i])):
                self.weights_input_to_hidden[i][j] += self.learning_rate * hidden_errors[i] * inputs[j]
            self.bias_hidden[i] += self.learning_rate * hidden_errors[i]

# Generación de datos para entrenamiento

def generate_line():
    """Genera una secuencia de puntos en línea recta en el plano xy."""
    x0, y0 = random.uniform(0, 10), random.uniform(0, 10)
    dx, dy = random.uniform(-1, 1), random.uniform(-1, 1)
    line_data = [[x0 + i * dx, y0 + i * dy] for i in range(10)]
    return [coord for point in line_data for coord in point]  # Vectoriza la lista de coordenadas

def generate_circular():
    """Genera una secuencia de puntos en movimiento circular en el plano xy."""
    radius = random.uniform(3, 5)
    center_x, center_y = random.uniform(4, 6), random.uniform(4, 6)
    circle_data = [[center_x + radius * math.cos(2 * math.pi * i / 10), center_y + radius * math.sin(2 * math.pi * i / 10)] for i in range(10)]
    return [coord for point in circle_data for coord in point]

def generate_random():
    """Genera una secuencia de puntos aleatorios en el plano xy."""
    random_data = [[random.uniform(0, 10), random.uniform(0, 10)] for _ in range(10)]
    return [coord for point in random_data for coord in point]

# Inicialización del perceptrón
perceptron = Perceptron(input_size=20, hidden_size=5, output_size=3, learning_rate=0.01)

# Entrenamiento del perceptrón
for epoch in range(50):  # Se puede ajustar el número de épocas
    for _ in range(100):  # 100 ejemplos de cada clase
        line_data = generate_line()
        perceptron.train(line_data, [1, 0, 0])  # Clase 0 para movimiento lineal

        circular_data = generate_circular()
        perceptron.train(circular_data, [0, 1, 0])  # Clase 1 para movimiento circular

        random_data = generate_random()
        perceptron.train(random_data, [0, 0, 1])  # Clase 2 para movimiento aleatorio

# Prueba del perceptrón
line_test = generate_line()
circular_test = generate_circular()
random_test = generate_random()

print(f"Predicción para una línea recta: {perceptron.predict(line_test)}")
print(f"Predicción para un movimiento circular: {perceptron.predict(circular_test)}")
print(f"Predicción para un movimiento aleatorio: {perceptron.predict(random_test)}")