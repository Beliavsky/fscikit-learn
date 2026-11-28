"""Conditioned KNN oracle: geometry bounds plus sklearn precomputed aggregation."""
import numpy as np
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor


def certify(model,native_fit,native_targets,native_queries,source_queries,proofs,y,classifier):
    source_fit=np.asarray(model._fit_X)
    np.testing.assert_allclose(native_fit,source_fit,rtol=3e-10,atol=3e-10)
    np.testing.assert_array_equal(native_targets.ravel(),np.asarray(y).ravel())
    n,p=native_fit.shape;k=model.n_neighbors;eps=np.finfo(float).eps
    forced=(KNeighborsClassifier if classifier else KNeighborsRegressor)(
        n_neighbors=k,weights='distance',algorithm='brute',metric='precomputed').fit(np.zeros((n,n)),y)
    native_geometry=(KNeighborsClassifier if classifier else KNeighborsRegressor)(
        n_neighbors=k,weights='distance',algorithm='brute').fit(native_fit,y)
    outputs=[];witnesses=[]
    for nq,sq,(dist,indices) in zip(native_queries,source_queries,proofs):
        indices=np.asarray(indices,dtype=int)-1
        assert dist.shape==indices.shape==(len(nq),k)
        assert np.all(indices>=0) and np.all(indices<n)
        assert all(len(set(row))==k for row in indices)
        native_delta=nq[:,None,:]-native_fit[None,:,:]
        native_d=np.sqrt(np.sum(native_delta*native_delta,axis=2))
        chosen=np.take_along_axis(native_d,indices,axis=1)
        np.testing.assert_array_equal(dist==0,chosen==0)
        # Source brute Euclidean uses squared-norm/dot cancellation. A relative
        # bound on the final distance against direct subtraction is invalid
        # near zero, even for Source's own distances. Check the actual Source
        # route exactly, plus its squared-distance forward error separately.
        source_native_dist,source_native_indices=native_geometry.kneighbors(nq,n_neighbors=n)
        source_native_full=np.empty_like(source_native_dist)
        np.put_along_axis(source_native_full,source_native_indices,source_native_dist,axis=1)
        source_native_chosen=np.take_along_axis(source_native_full,indices,axis=1)
        np.testing.assert_array_equal(dist,source_native_chosen)
        native_norm_bound=np.sum(nq[:,None,:]**2+native_fit[None,:,:]**2+
                                 2*np.abs(nq[:,None,:]*native_fit[None,:,:]),axis=2)
        native_selected_bound=np.take_along_axis(256*p*eps*native_norm_bound,indices,axis=1)
        native_squared_error=np.abs(dist*dist-chosen*chosen)
        assert np.all(native_squared_error<=native_selected_bound+np.finfo(float).tiny)
        legacy_relative_bound=bool(np.all(np.abs(dist-chosen)<=128*p*eps*np.abs(chosen)))
        kth=np.partition(native_d,k-1,axis=1)[:,k-1]
        assert np.all(chosen.max(axis=1)<=kth*(1+256*p*eps)),(chosen,kth)
        source_delta=sq[:,None,:]-source_fit[None,:,:]
        source_squared=np.sum(source_delta*source_delta,axis=2)
        source_d=np.sqrt(source_squared)
        perturb=np.linalg.norm(nq-sq,axis=1)+np.linalg.norm(native_fit-source_fit,axis=1).max(initial=0)
        source_kth=np.partition(source_d,k-1,axis=1)[:,k-1]
        source_chosen=np.take_along_axis(source_d,indices,axis=1).max(axis=1)
        assert np.all(source_chosen<=source_kth+2*perturb+256*p*eps*np.maximum(source_kth,1))
        sd,si=model.kneighbors(sq)
        square_selected=np.take_along_axis(source_squared,si,axis=1)
        norm_bound=np.sum(sq[:,None,:]**2+source_fit[None,:,:]**2+
                          2*np.abs(sq[:,None,:]*source_fit[None,:,:]),axis=2)
        kernel_bound=256*p*eps*norm_bound
        selected_bound=np.take_along_axis(kernel_bound,si,axis=1)
        kernel_error=np.abs(sd*sd-square_selected)
        assert np.all(kernel_error<=selected_bound+np.finfo(float).tiny)
        source_kth_squared=np.partition(source_squared,k-1,axis=1)[:,k-1]
        assert np.all(square_selected.max(axis=1)<=source_kth_squared+2*kernel_bound.max(axis=1))
        forced_dist=np.full((len(nq),n),max(1.,2*dist.max(initial=0)+1))
        np.put_along_axis(forced_dist,indices,dist,axis=1)
        pred=np.asarray(forced.predict(forced_dist)).reshape((len(nq),-1))
        proba=forced.predict_proba(forced_dist) if classifier else np.empty((0,0))
        outputs.append((pred,proba,forced_dist))
        witnesses.append(dict(changed_neighbor_sets=sum(set(a)!=set(b) for a,b in zip(indices,si)),
                              max_input_perturbation=float(perturb.max(initial=0)),
                              max_source_distance_squared_error=float(kernel_error.max(initial=0)),
                              max_source_distance_error_bound=float(selected_bound.max(initial=0)),
                              exact_source_native_distances=True,
                              legacy_native_relative_bound_passed=legacy_relative_bound,
                              max_native_distance_squared_error=float(native_squared_error.max(initial=0)),
                              max_native_distance_squared_error_bound=float(native_selected_bound.max(initial=0)),
                              max_native_distance_relative_error=float(np.max(
                                  np.abs(dist-chosen)/np.maximum(chosen,np.finfo(float).tiny),initial=0))))
    return forced,outputs,witnesses
