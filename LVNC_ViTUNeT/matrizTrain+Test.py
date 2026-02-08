import os
from loguru import logger
import typer
from utils import intercept_logger


app = typer.Typer()

@app.command()
@intercept_logger(logger=logger)
@logger.catch
def run(redEntrenada: str = typer.Option(..., exists=True, readable=True)):

    #EL FICHERO DEL TEST
    f = open(redEntrenada+"/test/informacion.txt","r")
    a = f.readlines()
    f.close()
    pacientesTest= a[2].split(" ")[2].replace("\n","")
    tn27Test = a[5].split(" ")[3].replace("\n","")
    fpPacientes27Test = a[6].split(":")[1].replace("\n","")[1:]
    fnPacientes27Test = a[7].split(":")[1].replace("\n","")[1:]
    tp27Test = a[8].split(" ")[3].replace("\n","")
    tn25Test = a[10].split(" ")[3].replace("\n","")
    fpPacientes25Test = a[11].split(":")[1].replace("\n","")[1:]
    fnPacientes25Test = a[12].split(":")[1].replace("\n","")[1:]
    tp25Test = a[13].split(" ")[3].replace("\n","")

    #EL FICHERO DEL ENTRENAMIENTO
    f = open(redEntrenada+"/matrizEntrenamientoVal.txt","r")
    a = f.readlines()
    f.close()
    pacientesTrain = a[0].split(":")[1].replace("\n","")[1:]
    tn27Train = a[2].split(" ")[3].replace("\n","")
    fpPacientes27Train = a[3].split(":")[1].replace("\n","")[1:]
    fnPacientes27Train = a[4].split(":")[1].replace("\n","")[1:]
    tp27Train = a[5].split(" ")[3].replace("\n","")
    tn25Train = a[7].split(" ")[3].replace("\n","")
    fpPacientes25Train = a[8].split(":")[1].replace("\n","")[1:]
    fnPacientes25Train = a[9].split(":")[1].replace("\n","")[1:]
    tp25Train = a[10].split(" ")[3].replace("\n","")

    tn27 = int(tn27Test) + int(tn27Train)
    fp27 = int(fpPacientes27Test.split(" ")[0]) + int(fpPacientes27Train.split(" ")[0])
    fn27 = int(fnPacientes27Test.split(" ")[0]) + int(fnPacientes27Train.split(" ")[0])
    tp27 = int(tp27Test) + int(tp27Train)

    tn25 = int(tn25Test) + int(tn25Train)
    fp25 = int(fpPacientes25Test.split(" ")[0]) + int(fpPacientes25Train.split(" ")[0])
    fn25 = int(fnPacientes25Test.split(" ")[0]) + int(fnPacientes25Train.split(" ")[0])
    tp25 = int(tp25Test) + int(tp25Train)

    FPpacientes27 = fpPacientes27Test[2:] + fpPacientes27Train[2:]
    FNpacientes27 = fnPacientes27Test[2:] + fnPacientes27Train[2:]
    FPpacientes25 = fpPacientes25Test[2:] + fpPacientes25Train[2:]
    FNpacientes25 = fnPacientes25Test[2:] + fnPacientes25Train[2:]


    with open(os.path.join(redEntrenada, "matrizTest+Train.txt"), "w") as f:
                    f.write(f"Pacientes totales: {int(pacientesTest) + int(pacientesTrain)}\n")
                    f.write(f'Para threshold 27,4 tenemos:\n\tPacientes sin enfermedad: {tn27}\n\tPacientes sin enfermedad pero diagnosticados enfermos: {fp27} {FPpacientes27} \n\tPacientes con enfermedad pero diagnosticados sanos: {fn27} {FNpacientes27}\n\tPacientes con enfermedad: {tp27}\n')
                    f.write(f'Para threshold 25 tenemos:\n\tPacientes sin enfermedad: {tn25}\n\tPacientes sin enfermedad pero diagnosticados enfermos: {fp25} {FPpacientes25}\n\tPacientes con enfermedad pero diagnosticados sanos: {fn25} {FNpacientes25}\n\tPacientes con enfermedad: {tp25}\n')
                    f.close()


if __name__=="__main__":
    app()
