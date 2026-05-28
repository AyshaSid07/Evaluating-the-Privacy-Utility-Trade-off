import org.deidentifier.arx.*;
import org.deidentifier.arx.AttributeType.Hierarchy;
import org.deidentifier.arx.criteria.KAnonymity;
import org.deidentifier.arx.criteria.DistinctLDiversity;
import org.deidentifier.arx.criteria.EqualDistanceTCloseness;
import java.nio.charset.StandardCharsets;

public class ARX_public_coverage {
    public static void main(String[] args) throws Exception {
        // path of the original income dataset
        String inputPath = "../datasets/folktables_public_coverage_RAW.csv";
        String outputBasePath = "../datasets/ARX_acs_public_coverage_"; 
        // load the dataset into ARX
        Data data = Data.create(inputPath, StandardCharsets.UTF_8, ',');
        // list of quasi identifiers used for anonymization
        String[] qis = {"AGEP", "SCHL", "SEX", "MAR", "ESP", "CIT", "MIG", "MIL", "ANC", "NATIVITY", "ESR"};
        // set all the attributes as insensitive attributes first
        for (int i = 0; i < data.getHandle().getNumColumns(); i++) {
            String colName = data.getHandle().getAttributeName(i);
            data.getDefinition().setAttributeType(colName, AttributeType.INSENSITIVE_ATTRIBUTE);
        }
        // load the hierarchy files for each quasi-identifier
        for (String qi : qis) {
            String hierarchyFileName = "hierarchies/" + qi.toLowerCase() + ".csv";
            data.getDefinition().setAttributeType(qi, Hierarchy.create(hierarchyFileName, StandardCharsets.UTF_8, ','));
        }
        // create ARX anonymizer object
        ARXAnonymizer anonymizer = new ARXAnonymizer();
        
        // k-anonymity loop
        int[] kValues = {3, 5, 10, 15};
        for (int k : kValues) {
            // create anonymization configuration
            ARXConfiguration config = ARXConfiguration.create();
            config.addPrivacyModel(new KAnonymity(k));            
            config.setSuppressionLimit(0.05d); 
            config.setAlgorithm(ARXConfiguration.AnonymizationAlgorithm.BEST_EFFORT_BOTTOM_UP);
            
            ARXResult result = anonymizer.anonymize(data, config);
            String out = outputBasePath + "k" + k + ".csv";
            result.getOutput(true).save(out, ',');
            data.getHandle().release();
        }
        // set the sensitive attribute RAC1P
        data.getDefinition().setAttributeType("RAC1P", AttributeType.SENSITIVE_ATTRIBUTE);
        // Base k value used for l-diversity and t-closeness
        int baseK = 5;
        
        // l-diversity loop
        int[] lValues = {3, 5};
        for (int l : lValues) {
            // create anonymization configuration
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
        
        // t-closeness loop
        double[] tValues = {0.15, 0.30};
        for (double t : tValues) {
            // create anonymization configuration
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