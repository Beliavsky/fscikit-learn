program reference_neighbor_outputs_driver
 use sklearn_common,only:dp
 use sklearn_neighbors,only:kneighbors_regressor,radius_neighbors_regressor
 implicit none
 type(kneighbors_regressor)::knn,fresh_k
 type(radius_neighbors_regressor)::rad,fresh_r
 real(dp),allocatable::x(:,:),y(:,:),q(:,:),truth(:,:),sw(:),prediction(:,:),matrix_prediction(:,:),single(:)
 real(dp)::radius,power,score
 integer::n,p,t,m,family,matrix,self_query,score_mode,score_rank,k,i,ios,case_id
 character(16)::metric,weights
 case_id=0
 do
  read(*,*,iostat=ios)n,p,t,m,family,matrix,self_query,score_mode,score_rank,k,radius,power,metric,weights
  if(ios/=0)exit
  knn=fresh_k;rad=fresh_r
  allocate(x(n,p),y(n,t),q(m,p),truth(m,t),sw(m))
  do i=1,n
   read(*,*)x(i,:),y(i,:)
  end do
  do i=1,m
   read(*,*)q(i,:),truth(i,:)
  end do
  read(*,*)sw
  knn%n_neighbors=k;knn%weights=weights;knn%metric=metric;knn%p=power
  rad%radius=radius;rad%weights=weights;rad%metric=metric;rad%p=power
  if(family==1)then
   if(matrix==1)then
    call knn%fit(x,y)
   else
    call knn%fit(x,y(:,1))
   end if
  else
   if(matrix==1)then
    call rad%fit(x,y)
   else
    call rad%fit(x,y(:,1))
   end if
  end if
  score=0
  if(self_query==2)then
   allocate(prediction(0,t),matrix_prediction(0,t))
  else if(family==1)then
   if(self_query==1)then
    matrix_prediction=knn%predict_matrix()
    if(matrix==1)then
     prediction=matrix_prediction
    else
     single=knn%predict();prediction=reshape(single,[n,1])
    end if
   else
    matrix_prediction=knn%predict_matrix(q)
    if(matrix==1)then
     prediction=matrix_prediction
    else
     single=knn%predict(q);prediction=reshape(single,[m,1])
    end if
   end if
   if(score_mode/=0)then
    if(self_query==1)then
     if(score_rank==1)then
      if(score_mode==2)then
       score=knn%score(y=truth,sample_weight=sw)
      else
       score=knn%score(y=truth)
      end if
     else
      if(score_mode==2)then
       score=knn%score(y=truth(:,1),sample_weight=sw)
      else
       score=knn%score(y=truth(:,1))
      end if
     end if
    else
     if(score_rank==1)then
      if(score_mode==2)then
       score=knn%score(q,truth,sample_weight=sw)
      else
       score=knn%score(q,truth)
      end if
     else
      if(score_mode==2)then
       score=knn%score(q,truth(:,1),sample_weight=sw)
      else
       score=knn%score(q,truth(:,1))
      end if
     end if
    end if
   end if
  else
   if(self_query==1)then
    matrix_prediction=rad%predict_matrix()
    if(matrix==1)then
     prediction=matrix_prediction
    else
     single=rad%predict();prediction=reshape(single,[n,1])
    end if
   else
    matrix_prediction=rad%predict_matrix(q)
    if(matrix==1)then
     prediction=matrix_prediction
    else
     single=rad%predict(q);prediction=reshape(single,[m,1])
    end if
   end if
   if(score_mode/=0)then
    if(self_query==1)then
     if(score_rank==1)then
      if(score_mode==2)then
       score=rad%score(y=truth,sample_weight=sw)
      else
       score=rad%score(y=truth)
      end if
     else
      if(score_mode==2)then
       score=rad%score(y=truth(:,1),sample_weight=sw)
      else
       score=rad%score(y=truth(:,1))
      end if
     end if
    else
     if(score_rank==1)then
      if(score_mode==2)then
       score=rad%score(q,truth,sample_weight=sw)
      else
       score=rad%score(q,truth)
      end if
     else
      if(score_mode==2)then
       score=rad%score(q,truth(:,1),sample_weight=sw)
      else
       score=rad%score(q,truth(:,1))
      end if
     end if
    end if
   end if
  end if
  case_id=case_id+1;write(*,'(a,i0)')'CASE ',case_id
  if(family==1)then
   write(*,*)knn%n_features_in_,knn%n_samples_fit_,knn%n_outputs_,merge(1,0,knn%matrix_target_)
   write(*,'(*(es25.16e3,1x))')knn%y_matrix_
  else
   write(*,*)rad%n_features_in_,rad%n_samples_fit_,rad%n_outputs_,merge(1,0,rad%matrix_target_)
   write(*,'(*(es25.16e3,1x))')rad%y_matrix_
  end if
  write(*,'(*(es25.16e3,1x))')prediction
  write(*,'(*(es25.16e3,1x))')matrix_prediction
  write(*,'(*(es25.16e3,1x))')score
  deallocate(x,y,q,truth,sw,prediction,matrix_prediction)
 end do
end program
