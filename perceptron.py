import math
import random

class Perceptron:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.001, seed=42):
        
        random.seed(seed)
        # Inicializamos los pesos y biases para la capa oculta
        self.weights_input_to_hidden = [[random.uniform(-1, 1) for _ in range(input_size)] for _ in range(hidden_size)]
        self.bias_hidden = [random.uniform(-1, 1)] * hidden_size
        
        # Inicializamos los pesos y biases para la capa de salida
        self.weights_hidden_to_output = [[random.uniform(-1, 1) for _ in range(hidden_size)] for _ in range(output_size)]
        self.bias_output = [random.uniform(-1, 1)] * output_size
        
        # Tasa de aprendizaje
        self.learning_rate = learning_rate

    def step_function(self, x):
        """Función escalón aplicada en la capa oculta."""
        return 1 if x >= 0 else 0

    def softmax(self, z):
        """Función softmax aplicada en la capa de salida."""
        exps = [math.exp(i) for i in z]
        sum_exps = sum(exps)
        return [exp_i / sum_exps for exp_i in exps]
    
    def capa_oculta(self, inputs):
        # Cálculo de la salida de la capa oculta
        self.hidden_layer_output = []
        for i in range(len(self.weights_input_to_hidden)):
            total = sum(w * inp for w, inp in zip(self.weights_input_to_hidden[i], inputs)) + self.bias_hidden[i]
            self.hidden_layer_output.append(self.step_function(total))
        return self.hidden_layer_output

    def capa_salida(self, hidden_layer_output):
        # Cálculo de la salida de la capa de salida
        self.output_layer_input = []
        for i in range(len(self.weights_hidden_to_output)):
            total = sum(w * hidden_out for w, hidden_out in zip(self.weights_hidden_to_output[i], hidden_layer_output)) + self.bias_output[i]
            self.output_layer_input.append(total)
        return self.output_layer_input

    def predict(self, inputs):
        """Predicción con una capa oculta y una capa de salida."""
        
        self.hidden_layer_output = self.capa_oculta(inputs)

        # Cálculo de la salida de la capa de salida
        self.output_layer_input = self.capa_salida(self.hidden_layer_output)
        
        # Aplicamos la función softmax en la capa de salida
        output = self.softmax(self.output_layer_input)
        
        # Devolvemos el índice de la clase con mayor probabilidad (1 para línea, 2 para línea circular)
        return 1 if output[0] > output[1] else 2

    def train(self, inputs, expected_output):
        """Entrenamiento con una capa oculta y una capa de salida."""

        # Hacemos el forward pass para calcular las salidas
        self.hidden_layer_output = self.capa_oculta(inputs)
        self.output_layer_input = self.capa_salida(self.hidden_layer_output)

        # Vector de salida esperada (1 para línea, 2 para círculo)
        expected_output_vector = [1, 0] if expected_output == 1 else [0, 1]
        
        # Cálculo del error en la capa de salida (antes de aplicar softmax)
        error_output = [expected - output for expected, output in zip(expected_output_vector, self.output_layer_input)]

        # Ajuste de los pesos y sesgos para la capa de salida
        for i in range(len(self.weights_hidden_to_output)):
            for j in range(len(self.weights_hidden_to_output[i])):
                # Actualización de los pesos con el error de salida
                self.weights_hidden_to_output[i][j] += self.learning_rate * error_output[i] * self.hidden_layer_output[j]
            # Actualización del sesgo de la capa de salida
            self.bias_output[i] += self.learning_rate * error_output[i]

        # Retropropagación del error a la capa oculta
        error_hidden = []
        for i in range(len(self.weights_input_to_hidden)):
            error = sum(self.weights_hidden_to_output[j][i] * error_output[j] for j in range(len(self.weights_hidden_to_output)))
            error_hidden.append(error)

        # Ajuste de los pesos y sesgos para la capa oculta
        for i in range(len(self.weights_input_to_hidden)):
            for j in range(len(self.weights_input_to_hidden[i])):
                # Actualización de los pesos de la capa oculta
                self.weights_input_to_hidden[i][j] += self.learning_rate * error_hidden[i] * inputs[j]
            # Actualización del sesgo de la capa oculta
            self.bias_hidden[i] += self.learning_rate * error_hidden[i]


def generate_line():
    """Genera una imagen de una línea horizontal o vertical en una matriz de 10x10."""
    image = [[0 for _ in range(10)] for _ in range(10)]
    if random.choice([True, False]):  # Horizontal
        row = random.randint(0, 9)
        for j in range(10):
            image[row][j] = 1
    else:  # Vertical
        col = random.randint(0, 9)
        for i in range(10):
            image[i][col] = 1
    return [pixel for row in image for pixel in row]

def generate_circular_line():
    """Genera una imagen de una circunferencia en una matriz de 10x10 con un radio y centro aleatorios."""
    image = [[0 for _ in range(10)] for _ in range(10)]

    # Generar un centro aleatorio para el círculo dentro de la matriz
    center = (random.randint(2, 7), random.randint(2, 7))  # Centro entre (2,7) para evitar bordes
    # Generar un radio aleatorio entre 2 y 4 píxeles
    radius = random.randint(2, 4)

    for i in range(10):
        for j in range(10):
            # Calcular la distancia al cuadrado desde el centro
            distance_squared = (i - center[0]) ** 2 + (j - center[1]) ** 2
            # Si está en el rango del radio, marcarlo como 1 (parte del círculo)
            if radius ** 2 - 1 <= distance_squared <= radius ** 2 + 1:
                image[i][j] = 1

    
    return [pixel for row in image for pixel in row]


perceptron = Perceptron(input_size=100, hidden_size=10, output_size=2, learning_rate=0.001)

# Entrenamiento 
for _ in range(10):  
    for _ in range(30):  
        line_image = generate_line()
        perceptron.train(line_image, 1)  # Línea recta tiene la etiqueta 1

    for _ in range(30):  
        circular_line_image = generate_circular_line()
        perceptron.train(circular_line_image, 2)  # Línea circular tiene la etiqueta 2

line_image = generate_line()
circular_line_image = generate_circular_line()

#Ejecución
print(f'Predicción para una línea recta: {perceptron.predict(line_image)}')
print(f'Predicción para una línea circular: {perceptron.predict(circular_line_image)}')
