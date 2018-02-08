package com.google.cloud.training.aslmlimmersion;

import java.io.Serializable;

import org.apache.beam.sdk.Pipeline;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.GroupByKey;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.values.KV;
import org.apache.beam.sdk.values.PCollection;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.google.cloud.training.aslmlimmersion.AddPrediction.MyOptions;

@SuppressWarnings("serial")
public abstract class InputOutput implements Serializable {
  protected static final Logger LOG = LoggerFactory.getLogger(InputOutput.class);
  public abstract PCollection<Baby> readInstances(Pipeline p, MyOptions options);
  public abstract void writePredictions(PCollection<Baby> instances, MyOptions options);

  private static class CreateBatch extends DoFn<Baby, KV<String, Baby>> {
    private static final int NUM_BATCHES = 5;
    @ProcessElement
    public void processElement(ProcessContext c) throws Exception {
      Baby f = c.element();
      String key = " " + (System.identityHashCode(f) % NUM_BATCHES);
      c.output(KV.of(key, f));
    }
  }
  
  public static PCollection<BabyPred> addPredictionInBatches(PCollection<Baby> instances) {
    // TODO: implement this method. you need to create a BabyPred for each Baby
    // see the addPredictionOneByOne method below. It will do the job, but
    // you want to improve performance by batching
    return null;
  }
  


  @SuppressWarnings("unused")
  private static PCollection<BabyPred> addPredictionOneByOne(PCollection<Baby> instances) {
    return instances //
        .apply("Inference", ParDo.of(new DoFn<Baby, BabyPred>() {
          @ProcessElement
          public void processElement(ProcessContext c) throws Exception {
            Baby f = c.element();
            double predwt = BabyweightMLService.predict(f, -5.0);
            c.output(new BabyPred(f, predwt));
          }
        }));
  }
}
