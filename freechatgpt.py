#pip install -U g4f[all]
import g4f

g4f.debug.logging = True  # Enable debug logging
g4f.debug.version_check = False  # Disable automatic version checking
print(g4f.Provider.Bing.params)  # Print supported args for Bing




# Using automatic a provider for the given model
## Streamed completion

response = g4f.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[{"role": "user", "content": "sprich deutsch mit mir"}],
    stream=True,
)
for message in response:
    print(message, flush=True, end='')


'''## Normal response
response = g4f.ChatCompletion.create(
    provider=g4f.Provider.Liaobots,
    model=g4f.models.gpt_4,
    messages=[{"role": "user", "content": "Ich hab nen langen ?"}],
)  # Alternative model setting

print(response)
'''
