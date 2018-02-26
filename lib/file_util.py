# a generator to read file in chunks
def read_chunks(reader, chunk_size=1000):
    chunk=[]
    for row in reader:
        chunk.append(row)
        if len(chunk) == chunk_size:
            yield chunk
            del chunk[:]
    yield chunk

# a method to write submission file in chunks
def write_chunks(writer, last_row_num, data):
    line_num = last_row_num
    for element in data:
        line_num += 1
        writer.writerow({'ImageId': line_num, 'Label': element})