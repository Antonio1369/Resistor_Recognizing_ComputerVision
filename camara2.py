def frames():
    copia = frame
    np_arr = tensor.cpu().detach().numpy()
    box = np.array([[np_arr[0][0], np_arr[0][3]], [np_arr[0][0], np_arr[0][1]], [np_arr[0][2], np_arr[0][1]], [np_arr[0][2],np_arr[0][3]]])
    box = np.int0(box)
    width = int(box[1][0])
    height = int(box[1][1])

    src_pts = box.astype('float32')
    dst_pts = np.array([[0, height-1],
                  [0,0],
                  [width-1,0],
                  [width-1,height-1]],dtype = 'float32')

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(copia, M, (width, height))
    hei, wid,_ = warped.shape
    if hei>wid:
        cont = ndimage.rotate(warped, 90)
    else:
        cont = warped.copy()

    #Guardamos la imagen
    cv2.imwrite("tempimg1.jpg", cont)

    #Resizeamos la imagen
    source = 'tempimg1.jpg'
    cont = Image.open(source)
    cont = cont.resize((200, 60)) #(width, height)
    cont.save(source)

    #Llamamos la imagen resizeada
    cont = cv2.imread(source)
    
    cont1 = cont.copy()
    he,w,_ = cont1.shape
    cont1 = cont1[int(he/2)-5:int(he/2)-3,:]
    print(cont1.shape)
    cont2 = cont.copy()
    h,w,_ = cont1.shape

    col=0
    sec = []
    lista = list()

    for i in range(w):
        seccion = cont1[:,(i):(i+1)]
        alto,ancho,_ = seccion.shape
    
        for k in range(3):
        
            for x in range(ancho):
                for y in range(alto):
                    col = seccion[y,x,k]/(ancho*alto) + col
            sec.append(int(col))
            col=0
        lista.append(sec)
        sec = []
    
    for i in range(w):
        for x in range(((i)),(i+1)):
            for y in range(he):
                cont2[y,x] = lista[i]
            
    cv2.imwrite("tempimg2.jpg", cont2)   
    imgT = Image.open(source)
    imgT = imgT.resize((200, 60)) #(width, height)
    imgT.save(source)
    imgT = cv2.imread(source)
    cv2.imwrite('rotacion.jpg',cont)

