from langchain_community.llms import CTransformers


# Singleton to ensure model is loaded only once
class LLMSingleton:
    _instance = None

    def __new__(cls):
        if cls._instance == None:
            print("Loading the LLM Model...")
            cls._instance = super(LLMSingleton, cls).__new__(cls)
            cls._instance.config = {'context_length': 10000,
                                    'repetition_penalty': 1.1, 
                                    'temperature': 0.3,
                                    'gpu_layers': 0,
                                    'max_new_tokens': 250,     # Ensures limit on the LLM output
                                    }
            
            cls._instance.model = CTransformers(model='yarn-mistral-7b-128k.Q4_K_M.gguf', 
                                                model_type='mistral', 
                                                config=cls._instance.config)
        return cls._instance
        
    def __call__(self, prompt):
        return self.model.invoke(prompt)
    