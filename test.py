import RAG
import cProfile
import pstats

def main():

    RAG.prepare_chunks(["Doc\Platon_critique_de_la_démocratie.pdf","Doc\Démocratie.pdf"])

if __name__ == "__main__":

    main()
