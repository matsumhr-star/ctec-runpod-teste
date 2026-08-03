import runpod

print("===================================")
print("CTEC RUNPOD TESTE")
print("Worker iniciado com sucesso!")
print("===================================")


def handler(job):
    print("Recebi um job:")
    print(job)

    return {
        "status": "ok",
        "message": "Servidor funcionando!",
        "input": job.get("input", {})
    }


if __name__ == "__main__":
    print("Iniciando RunPod Serverless...")
    runpod.serverless.start({"handler": handler})
