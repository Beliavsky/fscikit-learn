program reference_joint_classification_driver
 use sklearn_common,only:dp
 use sklearn_tree,only:decision_tree_classifier,extra_tree_classifier,tree_probability,tree_pruning_path
 implicit none
 class(decision_tree_classifier),allocatable::model
 type(tree_probability),allocatable::prob(:)
 type(tree_pruning_path)::path
 real(dp),allocatable::x(:,:),w(:),values(:,:),threshold(:),probabilities(:)
 integer,allocatable::y(:,:),prediction(:,:),classes(:)
 integer::n,p,q,mode,seed,depth,budget,nf,leaf,weighted,score_flag,i,k,first,total,status
 real(dp)::alpha,decrease
 character(64)::feature_token,argument
 real(dp)::feature_fraction
 logical::feature_metadata
 character(16)::criterion
 call get_command_argument(1,argument);feature_metadata=trim(argument)=='--features'
 do
  read(*,*,iostat=status)n,p,q,mode,criterion,seed,depth,budget,feature_token,leaf,alpha,decrease,weighted,score_flag
  if(status/=0)exit
  allocate(x(n,p),y(n,q),w(n))
  do i=1,n;read(*,*)x(i,:);end do
  do i=1,n;read(*,*)y(i,:);end do
  read(*,*)w
  if(mode==0)then
   allocate(decision_tree_classifier::model)
  else
   allocate(extra_tree_classifier::model)
  end if
  model%criterion=criterion;model%random_state=seed;model%max_depth=depth;model%max_leaf_nodes=budget
  model%min_samples_leaf=leaf;model%ccp_alpha=alpha;model%min_impurity_decrease=decrease
  select case(trim(feature_token))
  case('sqrt','log2','none');call model%set_max_features(trim(feature_token))
  case default
   if(scan(trim(feature_token),'.eEdD')>0)then
    read(feature_token,*)feature_fraction;call model%set_max_features(feature_fraction)
   else
    read(feature_token,*)nf;call model%set_max_features(nf)
   end if
  end select
  if(weighted==1)then
   call model%fit(x,y,w);path=model%cost_complexity_pruning_path(x,y,w)
  else
   call model%fit(x,y);path=model%cost_complexity_pruning_path(x,y)
  end if
  prediction=model%predict_multi(x);prob=model%predict_proba_multi(x);total=sum(model%n_classes_)
  allocate(values(model%node_count_,total),threshold(model%node_count_),classes(total),probabilities(n*total))
  do i=1,model%node_count_
   if(q==1)then
    values(i,:)=model%nodes_(i)%probability
   else
    values(i,:)=model%nodes_(i)%values
   end if
   threshold(i)=model%nodes_(i)%threshold
  end do
  first=1
  do k=1,q
   classes(first:first+model%n_classes_(k)-1)=model%classes_multi_(k)%values
   first=first+model%n_classes_(k)
  end do
  first=1
  do k=1,q
   probabilities(first:first+size(prob(k)%values)-1)=reshape(transpose(prob(k)%values),[size(prob(k)%values)])
   first=first+size(prob(k)%values)
  end do
  write(*,'(*(i0,1x))')model%node_count_,model%n_outputs_,model%n_features_in_,model%get_depth(),model%get_n_leaves()
  write(*,'(*(i0,1x))')model%n_classes_
  write(*,'(*(i0,1x))')classes
  write(*,'(*(i0,1x))')transpose(prediction)
  write(*,'(*(es26.17e3,1x))')probabilities
  write(*,'(*(i0,1x))')model%apply(x)
  write(*,'(*(es26.17e3,1x))')transpose(values)
  write(*,'(*(es26.17e3,1x))')model%nodes_(:model%node_count_)%impurity
  write(*,'(*(es26.17e3,1x))')model%nodes_(:model%node_count_)%weighted_n_samples
  write(*,'(*(i0,1x))')model%nodes_(:model%node_count_)%n_samples
  write(*,'(*(i0,1x))')model%nodes_(:model%node_count_)%left
  write(*,'(*(i0,1x))')model%nodes_(:model%node_count_)%right
  write(*,'(*(i0,1x))')model%nodes_(:model%node_count_)%feature
  write(*,'(*(es26.17e3,1x))')threshold
  write(*,'(*(es26.17e3,1x))')model%feature_importances_
  write(*,'(*(es26.17e3,1x))')path%ccp_alphas
  write(*,'(*(es26.17e3,1x))')path%impurities
  if(score_flag==1)then
   if(weighted==1)then
    write(*,'(*(es26.17e3,1x))')model%score(x,y,w)
   else
    write(*,'(*(es26.17e3,1x))')model%score(x,y)
   end if
  else
   write(*,*)
  end if
  if(feature_metadata)write(*,'(*(i0,1x))')model%max_features_
  deallocate(model,x,y,w,values,threshold,classes,probabilities)
 end do
end program
