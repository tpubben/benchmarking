import PhotoScan
from timeit import default_timer as timer

logfile = PhotoScan.app.getSaveFileName('save logfile as')
with open (logfile, 'a') as logging:
    logging.write("Begin Benchmarking \n\n")
projfile = PhotoScan.app.getOpenFileName('Select the project to benchmark')
doc = PhotoScan.app.document

def process_imagery(doc):
    doc.open(projfile)
    chunk = doc.chunk
    chunk.matchPhotos(accuracy=PhotoScan.HighAccuracy, generic_preselection=False, reference_preselection=True)
    chunk.alignCameras()
    chunk.buildDepthMaps(quality=PhotoScan.MediumQuality, filter=PhotoScan.AggressiveFiltering)
    chunk.buildDenseCloud()
    chunk.buildModel(surface=PhotoScan.Arbitrary, interpolation=PhotoScan.EnabledInterpolation)
    chunk.buildUV(mapping=PhotoScan.GenericMapping)
    chunk.buildTexture(blending=PhotoScan.MosaicBlending, size=4096)


# processor only

PhotoScan.app.gpu_mask = False
with open (logfile, 'a') as logging:
    logging.write("--- Processor Only ---\n\n")
for i in range(5):
    start = timer()
    process_imagery(doc)
    end = timer()
    with open(logfile, 'a') as logging:
        logging.write("Run number {}: {} seconds \r\n".format(str(i), str(end-start)))

# single GPU without Processor
PhotoScan.app.cpu_enable = False
PhotoScan.app.gpu_mask = True
with open(logfile, 'a') as logging:
    logging.write("--- Single GPU no CPU ---\n\n")
for i in range(5):
    start = timer()
    process_imagery(doc)
    end = timer()
    with open(logfile, 'a') as logging:
        logging.write("Run number {}: {} seconds \r\n".format(str(i), str(end - start)))

# dual GPU no Processor
PhotoScan.app.cpu_enable = False
PhotoScan.app.gpu_mask = 7
with open (logfile, 'a') as logging:
    logging.write("--- Dual GPU no CPU ---\n\n")
for i in range(5):
    start = timer()
    process_imagery(doc)
    end = timer()
    with open(logfile, 'a') as logging:
        logging.write("Run number {}: {} seconds \r\n".format(str(i), str(end-start)))
