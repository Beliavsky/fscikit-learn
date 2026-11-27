program reference_min_samples_class_forest_driver
 use sklearn_common,only:dp,unique_labels
 use sklearn_ensemble,only:random_forest_classifier,extra_trees_classifier
 use sklearn_tree,only:tree_probability
 implicit none
 character(64)::leaf_token,split_token
 integer::sample_count
 real(dp)::control_fraction
 integer::warm,reset
 class(random_forest_classifier),allocatable::model
 type(tree_probability),allocatable::prob(:),logs(:)
 real(dp),allocatable::x(:,:),w(:),values(:,:)
 integer,allocatable::y(:,:),labels(:),samples(:,:)
 integer::n,p,q,mode,seed,nt,boot,ns,oob,weighted,depth,budget,nf,leaf,cw,score_flag,status,i,j,k,first,total
 real(dp)::alpha,decrease
 character(64)::feature_token,argument
 real(dp)::feature_fraction
 logical::feature_metadata
 character(16)::criterion
 character(64)::sample_token
 real(dp)::sample_fraction
 call get_command_argument(1,argument);feature_metadata=trim(argument)=='--features'
 do
  read(*,*,iostat=status)n,p,q,mode,criterion,seed,nt,boot,sample_token,oob,weighted,depth,budget,feature_token,leaf_token,alpha,decrease,cw,score_flag,warm,reset,split_token
  if(status/=0)exit
  allocate(x(n,p),y(n,q),w(n))
  do i=1,n;read(*,*)x(i,:);end do
  do i=1,n;read(*,*)y(i,:);end do
  read(*,*)w
  if(reset==1.and.allocated(model))deallocate(model)
  if(.not.allocated(model))then
  if(mode==0)then
   allocate(random_forest_classifier::model)
  else
   allocate(extra_trees_classifier::model)
  end if
  end if
  model%warm_start=warm==1
  model%criterion=criterion;model%random_state=seed;model%n_estimators=nt
  call model%set_bootstrap(boot==1)
  if(scan(trim(sample_token),'.eEdD')>0)then
   read(sample_token,*)sample_fraction;call model%set_max_samples(sample_fraction)
  else
   read(sample_token,*)ns;call model%set_max_samples(ns)
  end if
  model%oob_score=oob==1;model%max_depth=depth;model%max_leaf_nodes=budget
  model%ccp_alpha=alpha;model%min_impurity_decrease=decrease
  model%class_weight='none'
  if(allocated(model%class_weight_values))deallocate(model%class_weight_values)
  if(allocated(model%class_weight_values_multi))deallocate(model%class_weight_values_multi)
  if(cw==1)model%class_weight='balanced'
  if(cw==2)model%class_weight='balanced_subsample'
  if(cw==3)then
   if(q==1)then
    allocate(model%class_weight_values);labels=unique_labels(y(:,1))
    model%class_weight_values%labels=labels
    model%class_weight_values%weights=[(real(j,dp)+.25_dp,j=1,size(labels))]
   else
    allocate(model%class_weight_values_multi(q))
    do k=1,q
     labels=unique_labels(y(:,k));model%class_weight_values_multi(k)%labels=labels
     model%class_weight_values_multi(k)%weights=[(real(j,dp)+.25_dp,j=1,size(labels))]
    end do
   end if
  end if
  if(scan(trim(leaf_token),'.eEdD')>0)then
   read(leaf_token,*)control_fraction;call model%set_min_samples_leaf(control_fraction)
  else
   read(leaf_token,*)sample_count;call model%set_min_samples_leaf(sample_count)
  end if
  if(scan(trim(split_token),'.eEdD')>0)then
   read(split_token,*)control_fraction;call model%set_min_samples_split(control_fraction)
  else
   read(split_token,*)sample_count;call model%set_min_samples_split(sample_count)
  end if
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
   call model%fit(x,y,w)
  else
   call model%fit(x,y)
  end if
  prob=model%predict_proba_multi(x);logs=model%predict_log_proba_multi(x)
  write(*,'(*(i0,1x))')model%n_features_in_,model%n_outputs_,size(model%estimators_)
  write(*,'(*(i0,1x))')model%n_classes_multi_
  write(*,'(*(i0,1x))')(model%classes_multi_(k)%values,k=1,q)
  write(*,'(*(i0,1x))')transpose(model%predict_multi(x))
  write(*,'(*(es26.17e3,1x))')(transpose(prob(k)%values),k=1,q)
  write(*,'(*(es26.17e3,1x))')(transpose(logs(k)%values),k=1,q)
  write(*,'(*(i0,1x))')transpose(model%apply(x))
  write(*,'(*(es26.17e3,1x))')model%feature_importances_
  write(*,'(*(i0,1x))')model%estimators_%random_state
  samples=model%estimators_samples();write(*,'(*(i0,1x))')transpose(samples)
  if(allocated(model%oob_counts_))then
   if(allocated(model%oob_decision_function_))then
    write(*,'(*(es26.17e3,1x))')transpose(model%oob_decision_function_)
   else
    write(*,'(*(es26.17e3,1x))')(transpose(model%oob_decision_function_multi_(k)%values),k=1,size(model%oob_decision_function_multi_))
   end if
   write(*,'(*(es26.17e3,1x))')model%oob_score_
   write(*,'(*(i0,1x))')model%oob_counts_
  else
   write(*,*);write(*,*);write(*,*)
  end if
  write(*,'(*(i0,1x))')(model%estimators_(i)%node_count_,model%estimators_(i)%get_depth(), &
   model%estimators_(i)%get_n_leaves(),model%estimators_(i)%n_features_in_,model%estimators_(i)%n_outputs_,i=1,nt)
  write(*,'(*(i0,1x))')(model%estimators_(i)%n_classes_,i=1,nt)
  write(*,'(*(i0,1x))')((model%estimators_(i)%classes_multi_(k)%values,k=1,q),i=1,nt)
  total=sum(model%n_classes_multi_)
  block
   real(dp),allocatable::packed(:)
   integer::at,a,b
   a=0
   do i=1,nt;a=a+model%estimators_(i)%node_count_*sum(model%estimators_(i)%n_classes_);end do
   allocate(packed(a));at=1
   do i=1,nt
    total=sum(model%estimators_(i)%n_classes_)
    do j=1,model%estimators_(i)%node_count_
     if(q==1)then
      packed(at:at+total-1)=model%estimators_(i)%nodes_(j)%probability
     else
      packed(at:at+total-1)=model%estimators_(i)%nodes_(j)%values
     end if
     at=at+total
    end do
   end do
   write(*,'(*(es26.17e3,1x))')packed
  end block
  write(*,'(*(es26.17e3,1x))')(model%estimators_(i)%nodes_(:model%estimators_(i)%node_count_)%impurity,i=1,nt)
  write(*,'(*(es26.17e3,1x))')(model%estimators_(i)%nodes_(:model%estimators_(i)%node_count_)%weighted_n_samples,i=1,nt)
  write(*,'(*(i0,1x))')(model%estimators_(i)%nodes_(:model%estimators_(i)%node_count_)%n_samples,i=1,nt)
  write(*,'(*(i0,1x))')(model%estimators_(i)%nodes_(:model%estimators_(i)%node_count_)%left,i=1,nt)
  write(*,'(*(i0,1x))')(model%estimators_(i)%nodes_(:model%estimators_(i)%node_count_)%right,i=1,nt)
  write(*,'(*(i0,1x))')(model%estimators_(i)%nodes_(:model%estimators_(i)%node_count_)%feature,i=1,nt)
  write(*,'(*(es26.17e3,1x))')(model%estimators_(i)%nodes_(:model%estimators_(i)%node_count_)%threshold,i=1,nt)
  write(*,'(*(es26.17e3,1x))')(model%estimators_(i)%feature_importances_,i=1,nt)
  if(score_flag==1)then
   if(weighted==1)then
    write(*,'(*(es26.17e3,1x))')model%score(x,y,w)
   else
    write(*,'(*(es26.17e3,1x))')model%score(x,y)
   end if
  else
   write(*,*)
  end if
  if(feature_metadata)write(*,'(*(i0,1x))')model%estimators_%max_features_
  deallocate(x,y,w)
 end do
end program
