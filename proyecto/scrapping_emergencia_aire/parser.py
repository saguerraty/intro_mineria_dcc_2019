import os
import csv
import slate3k as slate
import sys

print(list(range(5, 10)))

with open('/Users/juanpablosuazo/UChile/semestre10/mineriaDeDatos/hitos/hito2/reportes-cvs/pred.csv', 'w') as csvfile:
    filewriter = csv.writer(csvfile, delimiter=',')

    for i in range(5, 10):
        for filename in os.listdir('/Users/juanpablosuazo/UChile/semestre10/mineriaDeDatos/hitos/hito2/reportes/201' + str(i)):
            if filename.endswith(".pdf"):
                try:
                    with open('/Users/juanpablosuazo/UChile/semestre10/mineriaDeDatos/hitos/hito2/reportes/201' + str(i) + '/' + filename, 'rb') as f:
                        extracted_text = slate.PDF(f)

                    prediccion = str(extracted_text).replace("\\n", ' ').split('CALIDAD DEL AIRE PREVISTA PARA MAÑANA')
                    pred = prediccion[1].split('*')[0].split('CONDICIÓN')[0]
                    fecha = filename.split('.')[0].replace('Pronostico', '')

                    if fecha[0] == 'D':
                        fecha = fecha[22:]
                    if pred[4] == ' ':
                        pr = pred[5:].replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
                        filewriter.writerow([fecha, pr])
                    else:
                        pr = pred[4:].replace('  ', ' ').replace('  ', ' ').replace('  ', ' ')
                        filewriter.writerow([fecha, pr])
                except:
                    print(filename + ' ' + str(sys.exc_info()[0]))
