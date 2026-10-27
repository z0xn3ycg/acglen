import face_recognition
import cv2
import numpy as np

# 1. Carregar a imagem da pessoa autorizada e extrair o embedding
print("Carregando imagem de referência...")
imagem_referencia = face_recognition.load_image_file("usuario.jpg")
embeddings_referencia = face_recognition.face_encodings(imagem_referencia)

if len(embeddings_referencia) == 0:
    print("Erro: Nenhum rosto encontrado em 'usuario.jpg'. Adicione uma foto clara.")
    exit()

rosto_conhecido_embedding = embeddings_referencia[0]
nome_conhecido = "Autorizado"  # Você pode trocar pelo seu nome

# 2. Inicializar a webcam (0 geralmente é a webcam padrão do notebook/computador)
video_capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not video_capture.isOpened():
    print("Erro: Não foi possível acessar a webcam.")
    exit()

print("Pressione 'q' na janela do vídeo para sair.")

while True:
    # Capturar quadro por quadro da webcam
    ret, frame = video_capture.read()
    
    if not ret:
        print("Falha ao capturar imagem da webcam.")
        break

    # Reduzir o tamanho do quadro para acelerar o processamento
    frame_pequeno = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    
    # Converter a imagem de BGR (OpenCV) para RGB (face_recognition)
    rgb_frame = cv2.cvtColor(frame_pequeno, cv2.COLOR_BGR2RGB)

    # Encontrar todas as faces e embeddings no quadro atual da webcam
    localizacoes_faces = face_recognition.face_locations(rgb_frame)
    embeddings_faces = face_recognition.face_encodings(rgb_frame, localizacoes_faces)

    for (top, right, bottom, left), embedding_face in zip(localizacoes_faces, embeddings_faces):
        # Calcular a distância entre o rosto da webcam e o rosto de referência
        distancia = face_recognition.face_distance([rosto_conhecido_embedding], embedding_face)
        
        # Definir limiar (threshold). Abaixo de 0.6 indica que são a mesma pessoa.
        if distancia[0] <= 0.6:
            nome = nome_conhecido
            cor_retangulo = (0, 255, 0)  # Verde para autorizado
        else:
            nome = "Desconhecido"
            cor_retangulo = (0, 0, 255)  # Vermelho para desconhecido

        # Como reduzimos o quadro (fx=0.25), multiplicamos as coordenadas por 4
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Desenhar um retângulo ao redor do rosto
        cv2.rectangle(frame, (left, top), (right, bottom), cor_retangulo, 2)

        # Desenhar uma caixa com o nome abaixo do rosto
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), cor_retangulo, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, f"{nome} ({distancia[0]:.2f})", (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)

    # Exibir a imagem resultante na tela
    cv2.imshow('Reconhecimento Facial - Webcam', frame)

    # Pressione a tecla 'q' para sair do loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Liberar a webcam e fechar janelas do OpenCV
video_capture.release()
cv2.destroyAllWindows()