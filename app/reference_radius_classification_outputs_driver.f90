program reference_radius_classification_outputs_driver
 use sklearn_common,only:dp
 use sklearn_neighbors,only:radius_neighbors_classifier,neighbor_probability
 implicit none
 type(radius_neighbors_classifier)::model,fresh
 type(neighbor_probability),allocatable::prob(:)
 real(dp),allocatable::x(:,:),q(:,:),sw(:),single_prob(:,:)
 integer,allocatable::y(:,:),truth(:,:),prediction(:,:),single(:),outlabels(:)
 real(dp)::power,score,radius
 integer::n,p,t,m,matrix,self_query,score_mode,score_rank,outmode,i,j,ios,case_id
 character(16)::metric,weights
 case_id=0
 do
  read(*,*,iostat=ios)n,p,t,m,matrix,self_query,score_mode,score_rank,outmode,radius,power,metric,weights
  if(ios/=0)exit
  model=fresh;allocate(x(n,p),y(n,t),q(m,p),truth(m,t),sw(m),outlabels(t))
  do i=1,n
   read(*,*)x(i,:),y(i,:)
  end do
  do i=1,m
   read(*,*)q(i,:),truth(i,:)
  end do
  read(*,*)sw
  read(*,*)outlabels
  model%radius=radius;model%p=power;model%metric=metric;model%weights=weights
  select case(outmode)
  case(0);model%outlier_mode='error'
  case(1);model%outlier_mode='label';model%outlier_label=-99
  case(2);model%outlier_mode='label';model%outlier_labels=outlabels
  case(3);model%outlier_mode='most_frequent'
  end select
  if(matrix==1)then
   call model%fit(x,y)
  else
   call model%fit(x,y(:,1))
  end if
  score=0
  if(self_query==2)then
   allocate(prediction(0,t),single(0),single_prob(0,0),prob(0))
  else
   if(self_query==1)then
    prediction=model%predict_matrix();prob=model%predict_proba_multioutput()
   else
    prediction=model%predict_matrix(q);prob=model%predict_proba_multioutput(q)
   end if
   if(t==1)then
    if(self_query==1)then
     single=model%predict();single_prob=model%predict_proba()
    else
     single=model%predict(q);single_prob=model%predict_proba(q)
    end if
   else
    allocate(single(0),single_prob(0,0))
   end if
   if(score_mode/=0)then
    if(self_query==1)then
     if(score_rank==1)then
      if(score_mode==2)then
       score=model%score(y=truth,sample_weight=sw)
      else
       score=model%score(y=truth)
      end if
     else
      if(score_mode==2)then
       score=model%score(y=truth(:,1),sample_weight=sw)
      else
       score=model%score(y=truth(:,1))
      end if
     end if
    else
     if(score_rank==1)then
      if(score_mode==2)then
       score=model%score(q,truth,sample_weight=sw)
      else
       score=model%score(q,truth)
      end if
     else
      if(score_mode==2)then
       score=model%score(q,truth(:,1),sample_weight=sw)
      else
       score=model%score(q,truth(:,1))
      end if
     end if
    end if
   end if
  end if
  case_id=case_id+1;write(*,'(a,i0)')'CASE ',case_id
  write(*,'(*(i0,1x))')model%n_features_in_,model%n_samples_fit_,model%n_outputs_,merge(1,0,model%outputs_2d_)
  write(*,'(*(i0,1x))')(size(model%classes_multi_(j)%values),j=1,t)
  write(*,'(*(i0,1x))')(model%classes_multi_(j)%values,j=1,t)
  if(allocated(model%outlier_label_))then
   write(*,'(*(i0,1x))')model%outlier_label_
  else
   write(*,*)
  end if
  write(*,'(*(i0,1x))')model%y_matrix_
  write(*,'(*(i0,1x))')prediction
  write(*,'(*(i0,1x))')single
  do j=1,size(prob)
   write(*,'(*(es26.17e3,1x))',advance='no')prob(j)%values
  end do
  write(*,*)
  write(*,'(*(es26.17e3,1x))')single_prob
  write(*,'(es26.17e3)')score
  deallocate(x,y,q,truth,sw,prediction,single,single_prob,prob,outlabels)
 end do
end program
