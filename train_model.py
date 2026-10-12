"""Reproduce data repairs, chronological evaluation, and browser model export."""
import json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import clone
ROOT=Path(__file__).resolve().parents[1]
df=pd.read_csv(ROOT/'data/source.csv',dtype={'listing_id':str,'zip_code':str})
df=df.loc[:,~df.columns.str.startswith('Unnamed')].drop(columns=['Fireplaec','Fireplace'],errors='ignore')
for label,col in [('Stories','stories'),('Fireplaces','fireplaces')]:
 df[col]=pd.to_numeric(df.features.str.extract(label+r':\s*([^;]+)',expand=False),errors='coerce')
df['ac']=df.features.str.extract(r'Air conditioning:\s*([^;]+)',expand=False).str.strip().map({'Yes':1,'No':0})
df['listing_id']=df.listing_id.str.replace(' ','',regex=False)
land=pd.read_csv(ROOT/'data/land_area.csv',dtype={'listing_id':str});land.listing_id=land.listing_id.str.replace(' ','',regex=False)
df=df.merge(land,on='listing_id',how='left',validate='one_to_one')
df['sale_date']=pd.to_datetime(df.sale_date)
df['property_age']=df.sale_date.dt.year-df.year_built
# Restore exact semicolon-separated names and support categorical amenity filtering.
nearby=json.loads((ROOT/'data/nearby_provenance.json').read_text())
nearby_map={x['listing_id'].replace(' ',''):x['places'] for x in nearby}
df['nearby_places']=df.listing_id.map(lambda k:'; '.join(p['name'] for p in nearby_map.get(k,[])))
df['nearby_categories']=df.listing_id.map(lambda k:sorted(set(p['category'] for p in nearby_map.get(k,[]))))
df['nearest_metro_m']=df.listing_id.map(lambda k:min([p['distance_m'] for p in nearby_map.get(k,[]) if p['category']=='Metro station'],default=np.nan))
df.property_type=df.property_type.replace({'Town Inside':'Townhouse','Town End':'Townhouse'})
# Exclude the three ambiguous Multi records from prediction; keep them in analytics.
num=['sqft','land_sqft','bedrooms','bathrooms','property_age','stories','ac','fireplaces','latitude','longitude']
cat=['property_type','neighborhood','condition']
model_df=df[(df.property_type!='Multi')&(df.price>0)&(df.property_age>=0)].sort_values('sale_date').copy()
cut=model_df.sale_date.iloc[int(len(model_df)*.8)]
dev=model_df[model_df.sale_date<cut];test=model_df[model_df.sale_date>=cut]
pre=ColumnTransformer([('num',Pipeline([('impute',SimpleImputer(strategy='median')),('scale',StandardScaler())]),num),('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat)])
models={'Median baseline':DummyRegressor(strategy='median'),'Ridge':Ridge(alpha=20),'Random Forest':RandomForestRegressor(n_estimators=150,max_depth=15,min_samples_leaf=3,max_features=.8,random_state=42,n_jobs=2),'Gradient Boosting':GradientBoostingRegressor(n_estimators=240,learning_rate=.04,max_depth=3,min_samples_leaf=8,loss='huber',random_state=42)}
# Validation windows are grouped by date; no date overlaps a train/validation boundary.
dates=np.array(sorted(dev.sale_date.unique()));edges=[int(len(dates)*f) for f in [.4,.6,.8,1.0]]
cv={};oof={}
for name,est in models.items():
 errs=[];preds=[];actual=[]
 for j in range(3):
  boundary=pd.Timestamp(dates[edges[j]]);end=pd.Timestamp(dates[edges[j+1]]) if edges[j+1]<len(dates) else dev.sale_date.max()+pd.Timedelta(days=1)
  tr=dev[dev.sale_date<boundary];va=dev[(dev.sale_date>=boundary)&(dev.sale_date<end)]
  p=Pipeline([('pre',clone(pre)),('model',clone(est))])
  target=tr.price if name=='Median baseline' else np.log(tr.price)
  p.fit(tr[num+cat],target);v=p.predict(va[num+cat]);v=v if name=='Median baseline' else np.exp(v)
  errs.append(float(mean_absolute_error(va.price,v)));preds.extend(v);actual.extend(va.price)
 cv[name]=errs;oof[name]=np.abs(np.array(preds)-np.array(actual))
 print(name,'CV MAE',round(np.mean(errs)),flush=True)
best=min([n for n in models if n!='Median baseline'],key=lambda n:np.mean(cv[n]))
results=[];fitted={}
for name,est in models.items():
 p=Pipeline([('pre',clone(pre)),('model',clone(est))]);p.fit(dev[num+cat],dev.price if name=='Median baseline' else np.log(dev.price))
 pred=p.predict(test[num+cat]);pred=pred if name=='Median baseline' else np.exp(pred)
 results.append({'name':name,'cv_mae':float(np.mean(cv[name])),'fold_mae':cv[name],'test_mae':float(mean_absolute_error(test.price,pred)),'test_rmse':float(np.sqrt(mean_squared_error(test.price,pred))),'test_r2':float(r2_score(test.price,pred)),'test_median_error':float(np.median(np.abs(test.price-pred)))})
 fitted[name]=p
pipe=fitted[best];prep=pipe['pre'];est=pipe['model'];fn=prep.get_feature_names_out().tolist()
ns=prep.named_transformers_['num'];enc=prep.named_transformers_['cat']
def tree(t):
 s=t.tree_;return {'left':s.children_left.tolist(),'right':s.children_right.tolist(),'feature':s.feature.tolist(),'threshold':np.round(s.threshold,12).tolist(),'value':s.value[:,0,0].tolist()}
export={'name':best,'numeric':num,'categorical':cat,'medians':ns['impute'].statistics_.tolist(),'means':ns['scale'].mean_.tolist(),'scales':ns['scale'].scale_.tolist(),'categories':[x.tolist() for x in enc.categories_],'feature_names':fn,'error_band':float(np.quantile(oof[best],.8))}
if best=='Random Forest':export.update(kind='forest',trees=[tree(x) for x in est.estimators_])
elif best=='Gradient Boosting':export.update(kind='boosting',trees=[tree(x[0]) for x in est.estimators_],initial=float(est.init_.constant_[0,0]),learning_rate=est.learning_rate)
else:export.update(kind='ridge',coefficients=est.coef_.tolist(),intercept=float(est.intercept_))
imp=est.feature_importances_ if hasattr(est,'feature_importances_') else np.abs(est.coef_)
group={k:0. for k in num+cat}
for name,value in zip(fn,imp):
 raw=name.split('__',1)[1];key=raw if raw in num else next(k for k in cat if raw.startswith(k+'_'));group[key]+=float(value)
total=sum(group.values());importance=sorted([{'feature':k,'value':v/total} for k,v in group.items()],key=lambda x:-x['value'])
# Verify tree export independently against sklearn on a representative sample.
def browser_predict(row):
 vals=[float(row[k]) if pd.notna(row[k]) else export['medians'][i] for i,k in enumerate(num)]
 vals=[(v-export['means'][i])/export['scales'][i] for i,v in enumerate(vals)]
 for k,cs in zip(cat,export['categories']):vals += [float(row[k]==c) for c in cs]
 def walk(t):
  i=0
  while t['left'][i]!=-1:i=t['left'][i] if vals[t['feature'][i]]<=t['threshold'][i] else t['right'][i]
  return t['value'][i]
 if export['kind']=='forest':z=np.mean([walk(t) for t in export['trees']])
 elif export['kind']=='boosting':z=export['initial']+export['learning_rate']*sum(walk(t) for t in export['trees'])
 else:z=export['intercept']+np.dot(export['coefficients'],vals)
 return float(np.exp(z))
sample=test.iloc[::max(1,len(test)//40)].copy();reference=np.exp(pipe.predict(sample[num+cat]));exported=np.array([browser_predict(row) for _,row in sample.iterrows()]);assert np.allclose(reference,exported,rtol=1e-5)
(ROOT/'data/inference_checks.json').write_text(json.dumps([{'inputs':{k:(None if pd.isna(row[k]) else row[k]) for k in num+cat},'expected':float(p)} for (_,row),p in zip(sample.iterrows(),reference)]))
# Do not treat imputed renovation years as verified facts in cards.
record_cols=['listing_id','address','neighborhood','price','sqft','land_sqft','property_type','bedrooms','bathrooms','year_built','condition','latitude','longitude','zip_code','sale_date','stories','ac','fireplaces','property_age','nearby_places','nearby_categories','nearest_metro_m']
clean=df[record_cols].copy();clean['sale_date']=clean.sale_date.dt.strftime('%Y-%m-%d');clean=clean.replace({np.nan:None})
for c in clean.select_dtypes(include='number').columns:clean[c]=clean[c].map(lambda x:round(x,6) if isinstance(x,(float,int)) else x)
clean.to_csv(ROOT/'dist/homes.csv',index=False)
(ROOT/'dist/data.json').write_text(json.dumps(clean.to_dict('records'),allow_nan=False,separators=(',',':')))
(ROOT/'dist/model.json').write_text(json.dumps(export,separators=(',',':')))
report={'selected_model':best,'rows':len(df),'model_rows':len(model_df),'training_rows':len(dev),'test_rows':len(test),'cutoff':cut.strftime('%Y-%m-%d'),'date_min':df.sale_date.min().strftime('%Y-%m-%d'),'date_max':df.sale_date.max().strftime('%Y-%m-%d'),'results':results,'importance':importance,'export_verified_rows':len(sample),'export_max_difference':float(np.max(np.abs(reference-exported))),'method':'Three expanding chronological validation windows. Model selected using development CV MAE; final test untouched until evaluation. Learned imputers, scalers, and encoders fitted separately per fold. Models predict log sale price (except the median baseline).','limitations':['Current building snapshots can differ from characteristics at the sale date.','Qualified single-unit residential sales only; condos absent.','Assessment neighborhoods are not necessarily colloquial neighborhoods.','Nearby places are current DCGIS records, not verified active listings.','Prediction error band is an empirical validation residual band, not a guaranteed confidence interval.']}
(ROOT/'dist/report.json').write_text(json.dumps(report,indent=2));(ROOT/'data/cleaned.csv').write_text(clean.to_csv(index=False))
print(json.dumps({'selected':best,'results':results,'rows':len(df),'train':len(dev),'test':len(test),'verified_export':len(sample)},indent=2),flush=True)
