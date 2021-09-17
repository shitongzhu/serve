package org.pytorch.serve.wlm;

import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Iterator;
import org.pytorch.serve.job.Job;
import org.pytorch.serve.util.messages.BaseModelRequest;
import org.pytorch.serve.util.messages.ModelInferenceRequest;
import org.pytorch.serve.util.messages.ModelLoadModelRequest;
import org.pytorch.serve.util.messages.ModelWorkerResponse;
import org.pytorch.serve.util.messages.Predictions;
import org.pytorch.serve.util.messages.RequestInput;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class BatchAggregator {

    private static final Logger logger = LoggerFactory.getLogger(BatchAggregator.class);

    private Model model;
    private Map<String, Job> jobs;

    public BatchAggregator(Model model) {
        this.model = model;
        jobs = new LinkedHashMap<>();
    }

    public BaseModelRequest getRequest(String threadName, WorkerState state)
            throws InterruptedException {
        jobs.clear();

        ModelInferenceRequest req = new ModelInferenceRequest(model.getModelName());

        model.pollBatch(
                threadName, (state == WorkerState.WORKER_MODEL_LOADED) ? 0 : Long.MAX_VALUE, jobs);

        for (Job j : jobs.values()) {
            if (j.isControlCmd()) {
                if (jobs.size() > 1) {
                    throw new IllegalStateException(
                            "Received more than 1 control command. "
                                    + "Control messages should be processed/retrieved one at a time.");
                }
                RequestInput input = j.getPayload();
                int gpuId = -1;
                String gpu = input.getStringParameter("gpu");
                if (gpu != null) {
                    gpuId = Integer.parseInt(gpu);
                }
                return new ModelLoadModelRequest(model, gpuId);
            } else {
                j.setScheduled();
                req.addRequest(j.getPayload());
            }
        }
        return req;
    }

    public void sendResponse(ModelWorkerResponse message) {
        // TODO: Handle prediction level code

        if (message.getCode() == 200) {
            if (jobs.isEmpty()) {
                // this is from initial load.
                return;
            }
            Iterator<Predictions> predictionsIterator = message.getPredictions().iterator();

            while (predictionsIterator.hasNext()) {
                Predictions prediction = predictionsIterator.next();
                String jobId = prediction.getRequestId();
                Job job = jobs.get(jobId);

                if (job == null) {
                    throw new IllegalStateException("Unexpected job: " + jobId);
                }
                job.response(
                    prediction.getResp(),
                    prediction.getContentType(),
                    prediction.getStatusCode(),
                    prediction.getReasonPhrase(),
                    prediction.getHeaders());
            }
            
        } else {
            for (Map.Entry<String, Job> j : jobs.entrySet()) {
                
                if (j.getValue() == null) {
                    throw new IllegalStateException("Unexpected job: " + j.getKey());
                }
                j.getValue().sendError(message.getCode(), message.getMessage());
            }
        }
        jobs.clear();
    }

    public void sendError(BaseModelRequest message, String error, int status) {
        if (message instanceof ModelLoadModelRequest) {
            logger.warn("Load model failed: {}, error: {}", message.getModelName(), error);
            return;
        }

        if (message != null) {
            ModelInferenceRequest msg = (ModelInferenceRequest) message;
            Iterator<RequestInput> requestIterator = msg.getRequestBatch().iterator();
            while(requestIterator.hasNext()) {
                String requestId = requestIterator.next().getRequestId();
                Job job = jobs.get(requestId);

                if (job == null) {
                    logger.error("Unexpected job: " + requestId);
                } else {
                    job.sendError(status,error);
                }
            }

        } else {
            // Send the error message to all the jobs
            for (Map.Entry<String, Job> j : jobs.entrySet()) {
                String jobsId = j.getValue().getJobId();
                Job job = jobs.get(jobsId);

                if (job.isControlCmd()) {
                    job.sendError(status, error);
                } else {
                    // Data message can be handled by other workers.
                    // If batch has gone past its batch max delay timer?
                    model.addFirst(job);
                }
            }
        }
        jobs.clear();
    }
}
