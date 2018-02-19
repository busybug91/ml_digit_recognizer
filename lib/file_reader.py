# a generator to read file in chunks
def read_chunks(reader, chunk_size=1000):
    chunk=[]
    for row in reader:
        chunk.append(row)
        if len(chunk) == chunk_size:
            yield chunk
            del chunk[:]
    yield chunk

# TODO: Verify if chunks need to be converted to numpy array with elements as float