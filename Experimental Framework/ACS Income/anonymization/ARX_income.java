import org.deidentifier.arx.*;
import org.deidentifier.arx.AttributeType.Hierarchy;
import org.deidentifier.arx.criteria.KAnonymity;
import org.deidentifier.arx.criteria.DistinctLDiversity;
import org.deidentifier.arx.criteria.EqualDistanceTCloseness;
import java.nio.charset.StandardCharsets;

public class ARX_income {
    public static void main(String[] args) throws Exception {
        // Path to the 80% partitioned training dataset to prevent data leakage
        String inputPath = "../datasets/folktables_income_train.csv";
        String outputBasePath = "../datasets/ARX_acs_income_"; 
        
        // Load the dataset into ARX
        Data data = Data.create(inputPath, StandardCharsets.UTF_8, ',');
        
        // List of quasi identifiers used for anonymization
        String[] qis = {"AGEP", "SCHL", "SEX", "COW", "MAR", "RELP", "WKHP", "POBP", "OCCP"};
        
        // Set all the attributes as insensitive attributes first
        for (int i = 0; i < data.getHandle().getNumColumns(); i++) {
            String colName = data.getHandle().getAttributeName(i);
            data.getDefinition().setAttributeType(colName, AttributeType.INSENSITIVE_ATTRIBUTE);
        }
        
        // Load the hierarchy files for each quasi-identifier
        for (String qi : qis) {
            String hierarchyFileName = "hierarchies/" + qi.toLowerCase() + ".csv";
            data.getDefinition().setAttributeType(qi, Hierarchy.create(hierarchyFileName, StandardCharsets.UTF_8, ','));
        }
        
        // Create ARX anonymizer object
        ARXAnonymizer anonymizer = new ARXAnonymizer();
        
        // k-anonymity loop
        int[] kValues = {3, 5, 10, 15};
        for (int k : kValues) {
            ARXConfiguration config = ARXConfiguration.create();
            config.addPrivacyModel(new KAnonymity(k));            
            config.setSuppressionLimit(0.05d); 
            config.setAlgorithm(ARXConfiguration.AnonymizationAlgorithm.BEST_EFFORT_BOTTOM_UP);
            
            ARXResult result = anonymizer.anonymize(data, config);
            String out = outputBasePath + "k" + k + ".csv";
            result.getOutput(true).save(out, ',');
            data.getHandle().release();
        }
        
        // Set the sensitive attribute RAC1P
        data.getDefinition().setAttributeType("RAC1P", AttributeType.SENSITIVE_ATTRIBUTE);
        int baseK = 5;
        
        // l-diversity loop
        int[] lValues = {3, 5};
        for (int l : lValues) {
            ARXConfiguration configL = ARXConfiguration.create();
            configL.addPrivacyModel(new KAnonymity(baseK));
            configL.addPrivacyModel(new DistinctLDiversity("RAC1P", l));
            configL.setSuppressionLimit(0.05d);
            configL.setAlgorithm(ARXConfiguration.AnonymizationAlgorithm.BEST_EFFORT_BOTTOM_UP);

            ARXResult resultL = anonymizer.anonymize(data, configL);
            String out = outputBasePath + "k5_l" + l + ".csv";
            resultL.getOutput(true).save(out, ',');
            data.getHandle().release();
        }
        
        // t-closeness loop (0.3 instead of 0.30 for filename consistency)
        double[] tValues = {0.15, 0.3};
        for (double t : tValues) {
            ARXConfiguration configT = ARXConfiguration.create();
            configT.addPrivacyModel(new KAnonymity(baseK));
            configT.addPrivacyModel(new EqualDistanceTCloseness("RAC1P", t));
            configT.setSuppressionLimit(0.05d);
            configT.setAlgorithm(ARXConfiguration.AnonymizationAlgorithm.BEST_EFFORT_BOTTOM_UP);
            
            ARXResult resultT = anonymizer.anonymize(data, configT);
            String out = outputBasePath + "k5_t" + t + ".csv";
            resultT.getOutput(true).save(out, ',');
            data.getHandle().release();
        }
    }
}