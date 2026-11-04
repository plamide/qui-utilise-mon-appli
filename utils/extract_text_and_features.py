import re
import pandas as pd 
import numpy as np

def extract_features_and_text(df_act: pd.DataFrame):
        p_time   = re.compile(r"^t(\d{1,6})$")
        p_screen = re.compile(r"\(([^)]+)\)")
        p_module = re.compile(r"\$(\w+)\$")
        p_noise  = re.compile(r"\([^)]*\)|\$[^$]*\$|<[^>]*>")

        text_ecrans  = []
        text_modules = []
        text_actions = []
        numeric_features = []

        for row in df_act.values:
            t_max = 0
            num_actions = 0
            ecrans, modules, actions = [], [], []

            for val in row:
                if pd.isna(val):
                    continue
                val_str = str(val).strip()

                m_time = p_time.match(val_str)
                if m_time:
                    val_t = int(m_time.group(1))
                    if val_t > t_max:
                        t_max = val_t
                    continue

                num_actions += 1

                m_screen = p_screen.search(val_str)
                if m_screen:
                    ecrans.append(m_screen.group(1).replace(" ", "_"))

                for m_mod in p_module.finditer(val_str):
                    modules.append(m_mod.group(1))

                action_clean = p_noise.sub("", val_str).strip().replace(" ", "_")
                if action_clean:
                    actions.append(action_clean)

            text_ecrans.append(" ".join(ecrans))
            text_modules.append(" ".join(modules))
            text_actions.append(" ".join(actions))
            cadence = (num_actions / (t_max + 1)) * 60
            numeric_features.append([t_max, num_actions, cadence])

        return text_ecrans, text_modules, text_actions, np.array(numeric_features)
