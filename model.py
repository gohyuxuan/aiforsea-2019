import pickle


class Model:
    def __init__(self):
        self.model = pickle.load(open('gnb_model.pkl', 'rb'))
        print('Model successfully loaded.')

    def predict(self, data):
        return self.model.predict(data)


if __name__ == '__main__':
    Model()
