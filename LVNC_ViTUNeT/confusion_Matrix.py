import json
from sklearn.metrics import confusion_matrix
import os
import glob
import os
import pickle

from loguru import logger
import typer
from utils import intercept_logger
from pathlib import Path


app = typer.Typer()

@app.command()
@intercept_logger(logger=logger)
@logger.catch
def run(redEntrenada: str = typer.Option(..., exists=True, readable=True)):

    resultados_combinados = {"per_patient": []}

    def combinar_resultados(resultados_actuales, resultados_nuevos):
        # Combinar los resultados actuales con los nuevos
        for paciente in resultados_nuevos['per_patient']:
            resultados_actuales['per_patient'].append(paciente)
        return resultados_actuales

    def filtrar(lista, threshold):
        listaFiltrada = []
        for i in lista:
            if(float(i) <= threshold):
                listaFiltrada.append(0)
            else:
                listaFiltrada.append(1)
        return listaFiltrada
        
    def obtenerFPyFN(y_true,y_pred,pacientes):
        resultado = []
        falsoPositivo = []
        falsoNegativo = []
        for i in range(len(y_true)):
            if(y_true[i] != y_pred[i]):
                if (y_true[i] == 0):
                    falsoPositivo.append(pacientes[i])
                if (y_true[i] == 1):
                    falsoNegativo.append(pacientes[i])
        resultado.append(falsoPositivo)
        resultado.append(falsoNegativo)
        return resultado

    tn27 = 0
    fp27 = 0
    tp27 = 0
    fn27 = 0
    tn25 = 0
    fp25 = 0
    tp25 = 0
    fn25 = 0
    FPpacientes27 = []
    FNpacientes27 = []
    FPpacientes25 = []
    FNpacientes25 = []
    for i in range(5):
        #Calculamos la Matriz de confusion.
        with open(redEntrenada+f"/Fold{i}/test/_result_volumes.json") as f:
            data = json.load(f)
            f.close()
        
         # Combina los resultados de este archivo con los resultados acumulados
        resultados_combinados = combinar_resultados(resultados_combinados, data)

        y_pred = []
        y_true = []
        classes = []
        for paciente in data['per_patient']:
            y_pred.append(paciente['VT% computed'])
            y_true.append(paciente['VT%'])
            classes.append(paciente['patient'])
        y_pred27 = filtrar(y_pred,27.4)
        y_true27 = filtrar(y_true,27.4)

        y_pred25 = filtrar(y_pred,25)
        y_true25 = filtrar(y_true,25)


        pacientesFPyFN27 = obtenerFPyFN(y_true27,y_pred27,classes)
        pacientesFPyFN25 = obtenerFPyFN(y_true25,y_pred25,classes)

        FPpacientes27 = FPpacientes27 + pacientesFPyFN27[0]
        FNpacientes27 = FNpacientes27 + pacientesFPyFN27[1]
        FPpacientes25 = FPpacientes25 + pacientesFPyFN25[0]
        FNpacientes25 = FNpacientes25 + pacientesFPyFN25[1]

        cf_matrix27 = confusion_matrix(y_true27,y_pred27).ravel()
        cf_matrix25 = confusion_matrix(y_true25,y_pred25).ravel()

        tn27 = tn27 + cf_matrix27[0]
        fp27 = fp27 + cf_matrix27[1]
        fn27 = fn27 + cf_matrix27[2]
        tp27 = tp27 + cf_matrix27[3]

        tn25 = tn25 + cf_matrix25[0]
        fp25 = fp25 + cf_matrix25[1]
        fn25 = fn25 + cf_matrix25[2]
        tp25 = tp25 + cf_matrix25[3]

        #print(f'Para threshold 27,4 tenemos:\n\tPacientes sin enfermedad: {cf_matrix27[0]}\n\tPacientes sin enfermedad pero diagnosticados enfermos: {cf_matrix27[1]} {pacientesFPyFN27[0]}\n\tPacientes con enfermedad pero diagnosticados sanos: {cf_matrix27[2]} {pacientesFPyFN27[1]}\n\tPacientes con enfermedad: {cf_matrix27[3]}\n')
        #print(f'Para threshold 25 tenemos:\n\tPacientes sin enfermedad: {cf_matrix25[0]}\n\tPacientes sin enfermedad pero diagnosticados enfermos: {cf_matrix25[1]} {pacientesFPyFN25[0]}\n\tPacientes con enfermedad pero diagnosticados sanos: {cf_matrix25[2]} {pacientesFPyFN25[1]}\n\tPacientes con enfermedad: {cf_matrix25[3]}')

    #print(f'Para threshold 27,4 tenemos:\n\tPacientes sin enfermedad: {tn27}\n\tPacientes sin enfermedad pero diagnosticados enfermos: {fp27} {FPpacientes27} \n\tPacientes con enfermedad pero diagnosticados sanos: {fn27} {FNpacientes27}\n\tPacientes con enfermedad: {tp27}')
    #print(f'Para threshold 25 tenemos:\n\tPacientes sin enfermedad: {tn25}\n\tPacientes sin enfermedad pero diagnosticados enfermos: {fp25} {FPpacientes25}\n\tPacientes con enfermedad pero diagnosticados sanos: {fn25} {FNpacientes25}\n\tPacientes con enfermedad: {tp25}\n')
    with open(os.path.join(redEntrenada, "matrizEntrenamientoVal.txt"), "w") as f:
                f.write(f"Pacientes totales: {tn27+fp27+fn27+tp27}\n")
                f.write(f'Para threshold 27,4 tenemos:\n\tPacientes sin enfermedad: {tn27}\n\tPacientes sin enfermedad pero diagnosticados enfermos: {fp27} {FPpacientes27} \n\tPacientes con enfermedad pero diagnosticados sanos: {fn27} {FNpacientes27}\n\tPacientes con enfermedad: {tp27}\n')
                f.write(f'Para threshold 25 tenemos:\n\tPacientes sin enfermedad: {tn25}\n\tPacientes sin enfermedad pero diagnosticados enfermos: {fp25} {FPpacientes25}\n\tPacientes con enfermedad pero diagnosticados sanos: {fn25} {FNpacientes25}\n\tPacientes con enfermedad: {tp25}\n')
                f.close()


    # Guardar los resultados combinados en _all_results_volumes.json
    with open(os.path.join(redEntrenada, "_all_results_volumes.json"), "w") as f:
        json.dump(resultados_combinados, f)


if __name__=="__main__":
    app()
