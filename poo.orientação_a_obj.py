class veiculo:
    def movimentar(self):
        print(f'sou um veiculo e me desloco.')

    def __init__(self, fabricante, modelo):
        self.__fabricante = fabricante
        self.__modelo = modelo
        self.__num_registro = None

#setter
    def set_num_registro(self, registro):
        self.__num_registro = registro
        
#getter
    def get_fabri_modelo(self):
        print(f'modelo: {self.__modelo}, fabricante: {self.__fabricante}. \n')

    def get_num_registro(self):
        return self.__num_registro
    
class carro(veiculo):
    # metodo__init__ serpa herdado
    def movimentar(self):
        print(f'Sou um carro e ando pelas ruas')

class motocicleta(veiculo):
    def movimentar(self):
        print(f' Corro muito!!!')

class aviao(veiculo):
    def __init__(self, fabricante, modelo, categoria):
        self.__catgirl = categoria
        super().__init__(fabricante, modelo)

    def get_categoria(self):
        return self.__catgirl
    
    def movimentar(self):
        print('eu voo demais')

if __name__ == '__main__':
    # meu_veiculo = veiculo('GM', 'CADILLAC ESCALADE')
    # meu_veiculo.movimentar()
    # meu_veiculo.get_fabri_modelo()
    # meu_veiculo.set_num_registro('490321-1')
    # print(f'registro: {meu_veiculo.get_num_registro()}\n')

    # meu_camaro = carro('grandao', 'mega grandao')
    # meu_camaro.movimentar()
    # meu_camaro.get_fabri_modelo()

    # meu_camaro12 = carro(' loucao', 'mega loucao')
    # meu_camaro12.movimentar()
    # meu_camaro12.get_fabri_modelo()

    # moto = motocicleta('SUPER',  'MEGA MOTOS EXPLOSION')
    # moto.movimentar()
    # moto.get_fabri_modelo()

    megaaguia = aviao('jatoso', ' Jateado com cum modo 872', 'cat:grande e verde')
    megaaguia.movimentar()
    megaaguia.get_fabri_modelo()
    print(f'categoria: {megaaguia.get_categoria()}')

