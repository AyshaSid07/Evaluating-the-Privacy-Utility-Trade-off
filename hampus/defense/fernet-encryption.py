from cryptography.fernet import Fernet
import pandas as pd

df = pd.read_csv("../datasets/mendeley_data.csv", nrows = 4)

key = Fernet.generate_key()
cipher_suite = Fernet(key)

def encrypt_val(value, cipher):
    if pd.isna(value): # ignore NaN values
        return value
    
    val_str = str(value)
    token = cipher.encrypt(val_str.encode('utf-8'))
    return token.decode('utf-8')

def decrypt_val(value, cipher):
    if pd.isna(value):
        return value
    
    try:
        val_bytes = str(value).encode('utf-8')
        decrypted_message = cipher.decrypt(val_bytes)
        return decrypted_message.decode('utf-8')
    except Exception:
        return value
    
print(f"Before encryption: \n{df}")
df_encrypted = df.astype(object) 
df_encrypted = df_encrypted.map(lambda x: encrypt_val(x, cipher_suite))
print(f"After encryption: \n{df_encrypted}")
df_encrypted.to_csv("../datasets/fernet_encrypted.csv", index=False)
# df_decrypted = df_encrypted.map(lambda x: decrypt_val(x, cipher_suite))
# 
# for col in df_decrypted.columns:
    # try:
        # df_decrypted[col] = pd.to_numeric(df_decrypted[col])
    # except (ValueError, TypeError):
        # pass
# 
# print(f"After decryption: \n{df_decrypted}")
